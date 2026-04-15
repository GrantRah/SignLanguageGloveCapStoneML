#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

#define BUTTON_PIN 2

const unsigned long SAMPLE_INTERVAL = 30;

int gestureID = 1;

unsigned long lastSampleTime = 0;
bool isRecording = false;
bool buttonPreviouslyPressed = false;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Wire.begin();
  Wire.setClock(100000);

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void loop() {
  bool buttonPressed = (digitalRead(BUTTON_PIN) == LOW);
  unsigned long currentTime = millis();

  // =========================
  // START recording
  // =========================
  if (buttonPressed && !buttonPreviouslyPressed) {
    isRecording = true;
    lastSampleTime = 0;
  }

  // =========================
  // RECORD DATA
  // =========================
  if (isRecording && buttonPressed) {
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {

      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);

      Serial.print(gestureID);
      Serial.print(",");
      Serial.print(a.acceleration.x, 3);
      Serial.print(",");
      Serial.print(a.acceleration.y, 3);
      Serial.print(",");
      Serial.println(a.acceleration.z, 3);

      lastSampleTime = currentTime;
    }
  }

  // =========================
  // STOP recording + next gesture
  // =========================
  if (!buttonPressed && buttonPreviouslyPressed) {
    isRecording = false;
    gestureID++;
  }

  buttonPreviouslyPressed = buttonPressed;
}
