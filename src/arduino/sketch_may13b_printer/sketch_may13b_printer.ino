#include "Adafruit_Thermal.h"
#include "SoftwareSerial.h"

#define MAX_LINE_CHARS 32

#define TX_PIN_0 6 // Arduino transmit  YELLOW WIRE  labeled RX on printer
#define RX_PIN_0 5 // Arduino receive   GREEN WIRE   labeled TX on printer

#define TX_PIN_1 11 // Arduino transmit  YELLOW WIRE  labeled RX on printer
#define RX_PIN_1 10 // Arduino receive   GREEN WIRE   labeled TX on printer

SoftwareSerial mySerial0(RX_PIN_0, TX_PIN_0); // Declare SoftwareSerial obj first
Adafruit_Thermal printer0(&mySerial0);     // Pass addr to printer constructor

SoftwareSerial mySerial1(RX_PIN_1, TX_PIN_1); // Declare SoftwareSerial obj first
Adafruit_Thermal printer1(&mySerial1);     // Pass addr to printer constructor

void setup() {
  // pinMode(7, OUTPUT); digitalWrite(7, LOW);

  mySerial0.begin(9600);
  mySerial1.begin(9600);
  Serial.begin(9600);
  printer0.begin();
}

void loop() {
  if (Serial.available() > 0) {
    String s = Serial.readStringUntil('|');

    int PRINTER_IDX;
    switch (s[0]) {
      case '0':
        PRINTER_IDX = 0;
        break;
      case '1':
      default: // This one's much faster, so we'll use it as the main one!
        PRINTER_IDX = 1;
        break;
    }

    // for (unsigned int i = 0; i < s.length(); i += MAX_LINE_CHARS) {
    //   // First, find max num of characters we have left
    //   int numChars = (s.length() - i) < MAX_LINE_CHARS ? (s.length() - 1) : MAX_LINE_CHARS;
    //   String sub = s.substring(i, i + numChars);
    //   printer.println(sub);
    // }

    String buf = "";
    for (unsigned int i = 1; i < s.length(); i++) {
      if (s[i] == '\n') {
        (PRINTER_IDX == 0 ? printer0 : printer1).println(buf);
        delay(10);
        buf = "";
        continue;
      }
      buf += s[i];
      if (buf.length() >= MAX_LINE_CHARS) {
        (PRINTER_IDX == 0 ? printer0 : printer1).println(buf);
        delay(10);
        buf = "";
      }
    }
    
    if (buf.length() > 0) {
      (PRINTER_IDX == 0 ? printer0 : printer1).println(buf);
    }

    (PRINTER_IDX == 0 ? printer0 : printer1).feed(1);
    Serial.println("Sent");
  }

  delay(100);
}
