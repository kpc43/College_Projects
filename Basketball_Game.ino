#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Servo.h>

// Display setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET     1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Pins
const int buzzerPin = 8;
const int sensorPin = A1;
const int buttonPin = A5;
const int motorPin = 4;

Servo motor;

// Game state
bool gameActive = false;
unsigned long gameStartTime = 0;
int score = 0;
int lastSensorState = LOW;
int motorAngle = 60;
bool motorDir = true;
unsigned long lastMotorMove = 0;

void setup() {
  Serial.begin(9600);

  pinMode(buzzerPin, OUTPUT);
  pinMode(sensorPin, INPUT);
  pinMode(buttonPin, INPUT_PULLUP); // assuming button connects to GND

  motor.attach(motorPin);

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    while (1);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("Press Button to Start");
  display.display();
}

void loop() {
  if (!gameActive && digitalRead(buttonPin) == LOW) {
    startGame();
  }

  if (gameActive) {
    updateMotor();
    updateScore();
    updateDisplay();

    if (millis() - gameStartTime >= 45000) {
      endGame();
    }
  }
}

void startGame() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);

  for (int i = 3; i >= 1; i--) {
    display.clearDisplay();
    display.setCursor(50, 20);
    display.print(i);
    display.display();
    delay(1000);
  }

  display.clearDisplay();
  display.setCursor(10, 20);
  display.setTextSize(1);
  display.println("Game Started!");
  display.display();
  delay(1000);

  gameActive = true;
  score = 0;
  gameStartTime = millis();
  lastSensorState = LOW;
  lastMotorMove = millis();
}


void endGame() {
  gameActive = false;
  motor.write(90); // stop motor at neutral

  digitalWrite(buzzerPin, HIGH);
  delay(1000);
  digitalWrite(buzzerPin, LOW);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Game Over!");
  display.setCursor(0, 20);
  display.print("Score: ");
  display.println(score);
  display.setCursor(0, 40);
  display.println("Press to restart");
  display.display();
}

void updateMotor() {
  if (millis() - lastMotorMove >= 1000) {
    if (motorDir) {
      motorAngle = 60;
    } else {
      motorAngle = 120;
    }
    motor.write(motorAngle);
    motorDir = !motorDir;
    lastMotorMove = millis();
  }
}

void updateScore() {
  int sensorState = analogRead(sensorPin) > 500 ? HIGH : LOW;

  if (sensorState == HIGH && lastSensorState == LOW) {
    score++;
  }

  lastSensorState = sensorState;
}

void updateDisplay() {
  unsigned long timeLeft = 45 - (millis() - gameStartTime) / 1000;

  display.clearDisplay();
  display.setCursor(0, 0);
  display.print("Time: ");
  display.println(timeLeft);
  display.setCursor(0, 20);
  display.print("Score: ");
  display.println(score);
  display.display();
}