// Include the necessary libraries
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// Create an object for the MPU6050 sensor
Adafruit_MPU6050 mpu;

#define BUTTON_PIN 2  // Change to the pin your button is connected to

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("Serial connection ready");

  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Button with internal pull-up

  Wire.begin();
  Wire.setClock(100000);

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) delay(10);
  }

  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
}

void loop() {
  // Only read and send data if button is pressed
  if (digitalRead(BUTTON_PIN) == LOW) {  // Button pressed
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Print raw data (optional, keep for debugging)
    Serial.print("Acceleration: X:");
    Serial.print(a.acceleration.x);
    Serial.print(" Y:");
    Serial.print(a.acceleration.y);
    Serial.print(" Z:");
    Serial.println(a.acceleration.z);

    // Absolute acceleration values
    float ax = abs(a.acceleration.x);
    float ay = abs(a.acceleration.y);

    // Thresholds
    float thresholdX = 3.5;
    float thresholdY = 3.5;

    String output = "";  // NA by default

    // Strong Y only → 1
    if (ay > thresholdY && ax < thresholdX) {
      output = "1";
    }
    // Strong X only → 2
    else if (ax > thresholdX && ay <= thresholdY) {   // <- add ay <= thresholdY to avoid conflict
      output = "2";
    }
    // Both X and Y below thresholds → 3
    else if (ax <= thresholdX && ay <= thresholdY) {  // <- use <= to include exact threshold
      output = "3";
    }

    // Print only when a gesture is detected
    if (output != "") {
      Serial.print("GESTURE= 0");
      Serial.println(output);
    }

    Serial.println();
    delay(500);  // fast enough to catch motion
  }
}
