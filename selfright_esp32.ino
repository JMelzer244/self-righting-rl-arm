/*
  Runs the trained self-righting policy on the ESP32-S3.
  Idles until a flip (gyro spike), waits a beat, rights itself, then eases
  back to neutral and re-arms.

  Needs policy_weights.h (from export_policy.py), the ESP32Servo library,
  and an MPU6050 + two MG996R on their own 6V supply -- common ground with the ESP32.
*/

#include <Wire.h>
#include <ESP32Servo.h>
#include <math.h>
#include "policy_weights.h"

// ---- pins ----
#define I2C_SDA      14
#define I2C_SCL      13
#define SERVO0_PIN   12       // roll1
#define SERVO1_PIN   11       // roll2

// ---- control ----
const float CONTROL_HZ   = 50.0f;
const float DT           = 1.0f / CONTROL_HZ;     // 0.02 s
const float SERVO_SPEED  = 4.0f;                  // rad/s, MUST match sim SERVO_SPEED
const float JOINT_LIMIT  = 1.5707963f;            // ± rad, MUST match sim joint range
const float MAX_DSERVO   = SERVO_SPEED * DT;      // rad per tick (slew limit)

// ---- flip detection ----
const float GYRO_TILT_THRESH = .5f;   // rad/s; a flip spikes above this. Lower if not detected.
const unsigned long PAUSE_MS = 1000;  // wait 1s after a flip before righting

// ---- IMU axis mapping: chip frame -> sim body (part_1) frame ----
int   AX_SRC[3] = {0, 1, 2};
float AX_SGN[3] = {+1, +1, +1};
int   GY_SRC[3] = {0, 1, 2};
float GY_SGN[3] = {+1, +1, +1};

// ---- servo mapping: sim joint angle (rad) -> microseconds ----
int   CENTER_US[2]  = {1500, 1500};
float US_PER_RAD[2] = {637.0f, -637.0f};   // servo 2 direction flipped
int   SERVO_MIN_US  = 500;
int   SERVO_MAX_US  = 2500;

// ---- staged bring-up ----
#define ENABLE_SERVOS 1

const int MPU = 0x68;
Servo s0, s1;
float servo_rad[2] = {0.0f, 0.0f};
unsigned long next_tick = 0;

enum State { IDLE, WAIT, RIGHT, HOME };
State state = IDLE;
unsigned long pause_end = 0;

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}

void readIMU(float a[3], float g[3]) {
  Wire.beginTransmission(MPU); Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) { return; }       // bus error: skip this tick
  if (Wire.requestFrom(MPU, 14, true) != 14) { return; }  // short read: skip this tick
  int16_t ax = Wire.read() << 8 | Wire.read();
  int16_t ay = Wire.read() << 8 | Wire.read();
  int16_t az = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read();                       // skip temperature
  int16_t gx = Wire.read() << 8 | Wire.read();
  int16_t gy = Wire.read() << 8 | Wire.read();
  int16_t gz = Wire.read() << 8 | Wire.read();
  a[0] = ax / 16384.0f; a[1] = ay / 16384.0f; a[2] = az / 16384.0f;       // g units (±2g)
  const float D2R = 0.01745329f;
  g[0] = (gx / 131.0f) * D2R; g[1] = (gy / 131.0f) * D2R; g[2] = (gz / 131.0f) * D2R; // rad/s
}

void buildObs(float obs[8]) {
  float a[3], g[3]; readIMU(a, g);
  float n = sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
  if (n < 1e-6f) n = 1e-6f;
  float au[3] = {a[0]/n, a[1]/n, a[2]/n};
  for (int k = 0; k < 3; k++) obs[k]     = -AX_SGN[k] * au[AX_SRC[k]];
  for (int k = 0; k < 3; k++) obs[3 + k] =  GY_SGN[k] * g[GY_SRC[k]];
  obs[6] = servo_rad[0] / JOINT_LIMIT;
  obs[7] = servo_rad[1] / JOINT_LIMIT;
}

void policy(const float obs[8], float act[2]) {
  float h1[N_H1], h2[N_H2];
  for (int o = 0; o < N_H1; o++) {
    float s = L0_B[o];
    for (int i = 0; i < N_OBS; i++) s += L0_W[o][i] * obs[i];
    h1[o] = tanhf(s);
  }
  for (int o = 0; o < N_H2; o++) {
    float s = L1_B[o];
    for (int i = 0; i < N_H1; i++) s += L1_W[o][i] * h1[i];
    h2[o] = tanhf(s);
  }
  for (int o = 0; o < N_ACT; o++) {
    float s = L2_B[o];
    for (int i = 0; i < N_H2; i++) s += L2_W[o][i] * h2[i];
    act[o] = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);   // clip to action space
  }
}

void writeServo(Servo &s, int idx) {
  int us = CENTER_US[idx] + (int)(servo_rad[idx] * US_PER_RAD[idx]);
  if (us < SERVO_MIN_US) us = SERVO_MIN_US;
  if (us > SERVO_MAX_US) us = SERVO_MAX_US;
  s.writeMicroseconds(us);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  mpuWrite(0x6B, 0x00);   // wake MPU6050
  delay(100);
#if ENABLE_SERVOS
  s0.attach(SERVO0_PIN, SERVO_MIN_US, SERVO_MAX_US);
  s1.attach(SERVO1_PIN, SERVO_MIN_US, SERVO_MAX_US);
  writeServo(s0, 0); writeServo(s1, 1);
#endif
  Serial.println("armed -- waiting for a flip");
  next_tick = millis();
}

void loop() {
  if (millis() < next_tick) return;
  next_tick += (unsigned long)(DT * 1000.0f);

  float obs[8], act[2];
  buildObs(obs);
  float gmag = sqrtf(obs[3]*obs[3] + obs[4]*obs[4] + obs[5]*obs[5]);  // rotation rate
  float up   = -obs[2];                                               // +1 upright, -1 flipped

  switch (state) {
    case IDLE:
      if (gmag > GYRO_TILT_THRESH) {
        state = WAIT;
        pause_end = millis() + PAUSE_MS;
        Serial.println("flip detected -> pausing 1s");
      }
      break;

    case WAIT:
      if (gmag > GYRO_TILT_THRESH) pause_end = millis() + PAUSE_MS;   // still moving: extend
      if (millis() >= pause_end) {
        state = RIGHT;
        Serial.println("righting");
      }
      break;

    case RIGHT:
      policy(obs, act);
      for (int j = 0; j < 2; j++) {
        float desired = act[j] * JOINT_LIMIT;
        float d = desired - servo_rad[j];
        if (d >  MAX_DSERVO) d =  MAX_DSERVO;
        if (d < -MAX_DSERVO) d = -MAX_DSERVO;
        servo_rad[j] += d;
      }
      if (up > 0.95f && gmag < 0.5f) {   // settled upright -> head home
        state = HOME;
        Serial.println("upright -> returning to neutral");
      }
      break;

    case HOME: {
      bool settled = true;
      for (int j = 0; j < 2; j++) {
        float d = 0.0f - servo_rad[j];                 // neutral is straight up
        if (d >  MAX_DSERVO) d =  MAX_DSERVO;           // same slew limit as righting
        if (d < -MAX_DSERVO) d = -MAX_DSERVO;
        servo_rad[j] += d;
        if (fabsf(servo_rad[j]) > 1e-3f) settled = false;
      }
      if (settled) {
        servo_rad[0] = 0.0f; servo_rad[1] = 0.0f;       // snap to exact center
        state = IDLE;
        Serial.println("neutral -> armed");
      }
      break;
    }
  }

#if ENABLE_SERVOS
  writeServo(s0, 0);
  writeServo(s1, 1);
#endif
}