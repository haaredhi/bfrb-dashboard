#include <Haaredhi-project-1_inferencing.h>
#include <Wire.h>
#include "MAX30105.h"
#include <CodeCell.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>

// ─────────────────────────────────────────────
//  Wi-Fi & Firebase configuration
// ─────────────────────────────────────────────
const char* WIFI_SSID    = "MUHAIF-4G";
const char* WIFI_PASSWORD = "aim210484";
const char* FIREBASE_URL  = "https://bfrb-dashboard-default-rtdb.firebaseio.com/events.json";
const char* NTP_SERVER    = "pool.ntp.org";
const long  GMT_OFFSET_S  = 3 * 3600;
const int   DST_OFFSET_S  = 0;

// ─────────────────────────────────────────────
//  Hardware
// ─────────────────────────────────────────────
MAX30105 hrSensor;
CodeCell myCodeCell;

#define GSR_PIN       1
#define VIBRATOR_PIN  5

const float ADC_VREF = 3.3f;
const int   ADC_MAX  = 4095;

bool hrReady = false;

float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;

float         irFiltered   = 0;
float         prevIR       = 0;
unsigned long lastBeatTime = 0;
float         bpmSmooth    = 0;

const long  fingerThreshold = 50000;
const float alphaIR         = 0.1f;
const float changeThreshold = 200.0f;

// ─────────────────────────────────────────────
//  Inference buffer
// ─────────────────────────────────────────────
static float sampleBuffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static int   sampleIndex = 0;

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 20;

// ─────────────────────────────────────────────
//  Vibration
// ─────────────────────────────────────────────
const unsigned long VIBRATE_DURATION_MS = 500;
unsigned long vibrateStartTime = 0;
bool          isVibrating      = false;

// ─────────────────────────────────────────────
//  Classification
// ─────────────────────────────────────────────
const float NAIL_BITING_THRESHOLD = 0.85f;
#define LABEL_HAND_TO_FACE  0
#define LABEL_NAIL_BITING   1
#define LABEL_REST          2

// ─────────────────────────────────────────────
//  Non-blocking Firebase POST flag
//  Set inside runInference() after vibration is triggered.
//  Executed in loop() so the HTTP request never blocks
//  the sampling loop or delays the next inference window.
// ─────────────────────────────────────────────
bool   pendingPost = false;
String pendingTs   = "";
float  pendingConf = 0.0f;
int    pendingHr   = -1;
int    pendingWd   = -1;

void runInference();

// ─────────────────────────────────────────────
//  Time helpers
// ─────────────────────────────────────────────
String getNowString() {
  struct tm ti;
  int retries = 0;
  while (!getLocalTime(&ti) && retries < 5) { delay(200); retries++; }
  if (retries >= 5) return "Time unavailable";
  char buf[40];
  strftime(buf, sizeof(buf), "%a %d %b %Y  %H:%M:%S", &ti);
  return String(buf);
}

int getNowHour() {
  struct tm ti;
  int retries = 0;
  while (!getLocalTime(&ti) && retries < 5) { delay(200); retries++; }
  if (retries >= 5) return -1;
  return ti.tm_hour;
}

int getNowWeekday() {
  struct tm ti;
  int retries = 0;
  while (!getLocalTime(&ti) && retries < 5) { delay(200); retries++; }
  if (retries >= 5) return -1;
  return ti.tm_wday;
}

// ─────────────────────────────────────────────
//  Firebase POST
// ─────────────────────────────────────────────
void postToFirebase(String timestamp, float confidence, int hour, int weekday) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[FIREBASE] Wi-Fi not connected — skipping upload");
    return;
  }

  const char* dayNames[] = {"Sunday","Monday","Tuesday","Wednesday",
                             "Thursday","Friday","Saturday"};
  String dayName = (weekday >= 0 && weekday < 7) ? dayNames[weekday] : "Unknown";

  String payload = "{";
  payload += "\"timestamp\":\"" + timestamp + "\",";
  payload += "\"label\":\"nail_biting\",";
  payload += "\"confidence\":"  + String(confidence * 100, 1) + ",";
  payload += "\"hour\":"        + String(hour) + ",";
  payload += "\"weekday\":\""   + dayName + "\"";
  payload += "}";

  HTTPClient http;
  http.begin(FIREBASE_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  if (code > 0) {
    Serial.print("[FIREBASE] Posted — HTTP "); Serial.println(code);
  } else {
    Serial.print("[FIREBASE] Failed — "); Serial.println(http.errorToString(code));
  }
  http.end();
}

// ─────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(VIBRATOR_PIN, OUTPUT);
  digitalWrite(VIBRATOR_PIN, LOW);
  analogReadResolution(12);

  Wire.begin(8, 9);
  Wire.setClock(100000);

  if (hrSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    hrSensor.setup();
    hrSensor.setPulseAmplitudeRed(0x3F);
    hrSensor.setPulseAmplitudeIR(0x3F);
    hrSensor.setPulseAmplitudeGreen(0);
    hrReady = true;
    Serial.println("[OK] HR sensor ready");
  } else {
    hrReady = false;
    Serial.println("[WARN] HR sensor not found");
  }

  myCodeCell.Init(MOTION_ACCELEROMETER | MOTION_GYRO);
  Serial.println("[OK] IMU ready");

  // ── Connect to Wi-Fi ──
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500); Serial.print("."); tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[OK] Wi-Fi connected");
    configTime(GMT_OFFSET_S, DST_OFFSET_S, NTP_SERVER);
    delay(3000);
    Serial.println("[OK] NTP time synced");
    Serial.println("[OK] Firebase cloud logging ready");
  } else {
    Serial.println("\n[WARN] Wi-Fi failed — running offline, Firebase disabled");
  }

  Serial.println("[OK] BFRB Detection running...");
}

// ─────────────────────────────────────────────
//  Main Loop
// ─────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // ── Non-blocking vibration timer ──
  if (isVibrating && (now - vibrateStartTime >= VIBRATE_DURATION_MS)) {
    digitalWrite(VIBRATOR_PIN, LOW);
    isVibrating = false;
  }

  // ── Non-blocking Firebase POST ──
  // Executed here so the HTTP request never delays vibration
  // or the next inference window.
  if (pendingPost) {
    pendingPost = false;
    postToFirebase(pendingTs, pendingConf, pendingHr, pendingWd);
  }

  // ── 20 ms sampling gate ──
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // ── IMU ──
  if (myCodeCell.Run(1)) {
    myCodeCell.Motion_AccelerometerRead(ax, ay, az);
    myCodeCell.Motion_GyroRead(gx, gy, gz);
  }

  // ── Heart rate ──
  if (hrReady) {
    long ir = hrSensor.getIR();
    if (ir < fingerThreshold) {
      bpmSmooth = 0; irFiltered = 0; prevIR = 0;
    } else {
      if (irFiltered == 0) irFiltered = (float)ir;
      irFiltered = alphaIR * ir + (1.0f - alphaIR) * irFiltered;
      if (irFiltered > prevIR + changeThreshold) {
        unsigned long dt = now - lastBeatTime;
        if (dt > 400 && dt < 1500) {
          float bpmRaw = 60000.0f / (float)dt;
          bpmSmooth = (bpmSmooth == 0) ? bpmRaw : 0.9f * bpmSmooth + 0.1f * bpmRaw;
        }
        lastBeatTime = now;
      }
      prevIR = irFiltered;
    }
  }

  // ── GSR ──
  int   gsrRaw     = analogRead(GSR_PIN);
  float gsrVoltage = (float)gsrRaw * (ADC_VREF / (float)ADC_MAX);

  // ── Fill sample buffer ──
  int base = sampleIndex * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
  sampleBuffer[base + 0] = ax;
  sampleBuffer[base + 1] = ay;
  sampleBuffer[base + 2] = az;
  sampleBuffer[base + 3] = gx;
  sampleBuffer[base + 4] = gy;
  sampleBuffer[base + 5] = gz;
  sampleBuffer[base + 6] = bpmSmooth;
  sampleBuffer[base + 7] = gsrVoltage;
  sampleIndex++;

  // ── Run inference once buffer is full (every 2 s / 98 samples) ──
  if (sampleIndex >= EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
    sampleIndex = 0;
    Serial.print("AX:"); Serial.print(ax);
    Serial.print(" AY:"); Serial.print(ay);
    Serial.print(" AZ:"); Serial.println(az);
    Serial.print("GX:"); Serial.print(gx);
    Serial.print(" GY:"); Serial.print(gy);
    Serial.print(" GZ:"); Serial.println(gz);
    Serial.print("HR:"); Serial.print(bpmSmooth);
    Serial.print(" GSR:"); Serial.println(gsrVoltage);
    runInference();
  }
}

// ─────────────────────────────────────────────
//  Inference + Feedback
// ─────────────────────────────────────────────
void runInference() {
  signal_t signal;
  numpy::signal_from_buffer(sampleBuffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.print("[ERROR] "); Serial.println(err);
    return;
  }

  Serial.println("-----------------------------");
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    Serial.print("  ");
    Serial.print(result.classification[i].label);
    Serial.print(": ");
    Serial.println(result.classification[i].value, 3);
  }

  // ── Find top class ──
  int   topClass = 0;
  float topScore = 0;
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > topScore) {
      topScore = result.classification[i].value;
      topClass = i;
    }
  }
  Serial.print("  -> ");
  Serial.print(result.classification[topClass].label);
  Serial.print(" ("); Serial.print(topScore * 100, 1); Serial.println("%)");

  // ── Nail biting detected above confidence threshold ──
  if (result.classification[LABEL_NAIL_BITING].value >= NAIL_BITING_THRESHOLD) {
    Serial.println("  !! NAIL BITING DETECTED - vibrating!");

    // ── Vibrate FIRST for instant haptic feedback ──
    if (!isVibrating) {
      digitalWrite(VIBRATOR_PIN, HIGH);
      vibrateStartTime = millis();
      isVibrating = true;
    }

    // ── Schedule Firebase POST (executed in loop() — non-blocking) ──
    pendingPost = true;
    pendingTs   = getNowString();
    pendingConf = result.classification[LABEL_NAIL_BITING].value;
    pendingHr   = getNowHour();
    pendingWd   = getNowWeekday();
  }

  Serial.println("-----------------------------");
}
