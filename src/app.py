from uuid import uuid4
import time
from random import random
import math
import json
from flask import Flask, render_template, request
import ngrams.ngrams as ngrams
import os
import openai
import sys
import serial
from func_timeout import func_timeout, FunctionTimedOut

# -- PARAMS -- #

QUESTIONS = [
	"Share one creative thing you saw today or recently that you found interesting.",
	"Think about your own experiences with creative expression, perhaps focusing on one medium or activity. With this in mind, describe the literal or technical steps you take in your creative process.",
	"Consider how you feel while engaging in that creative process. Write a few ambiguous steps someone else could interpret to retrace the experience you have while creating in a medium of your choice."
]

USE_PREEXISTING_FILE = None # "data/MAIN_icwc_1683672331.json" # "data/icwc_1683672331.json" # None or filename

SERIAL_PORT = "/dev/cu.usbmodem2101"
SERIAL_BAUD_RATE = 9600

# -- UTILS & MAIN LOGIC -- #

filename = f"data/icwc_{round(time.time())}.json" if USE_PREEXISTING_FILE is None else USE_PREEXISTING_FILE
responses = dict()
ngramsModel = ngrams.NGramModel(lambda x: x.split(" "), lambda x: " ".join(x), 2)

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

ser = None

# JSON Data

def readData():
	global responses
	with open(filename, "r") as infile:
		responses = json.load(infile)

def saveData():
	global responses
	with open(filename, "w") as outfile:
		json.dump(responses, outfile)

def storeResponse(participantId, text):
	global responses
	if participantId not in responses:
		responses[participantId] = []
	responses[participantId].append(text)

# Printing

def sendToPrinter(text, printerIdx):
	# TODO: Vary by printerIdx
	if not ser.is_open:
		ser.open()
		time.sleep(1)
	ser.write((f"{printerIdx}{text}|").encode("UTF-8"))

def sendToPrinterWithHandshake(text, printerIdx, timeout=10):
	sendToPrinter(text, printerIdx)
	print(f"Sending text to {printerIdx}: {text}")
	sendToPrinter(text, printerIdx)
	
	try:
		result = func_timeout(timeout, ser.readline)
		print(result)
		return True
	except FunctionTimedOut:
		print("Failed to send: timed out.")
		return False

# Mangling Text

def mangleWithGPT(text):
	completion = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "user", "content": f"Imagine that you are a surrealist creative writer. Someone provides you the following instructions for how to write a very short, surreal, fragmented, choppy, ambiguous piece of provocative prose: \"{text}\" What do you write?"}
		]
	)
	return completion.choices[0].message

def addToNGrams(participantId, questionIdx, text):
	ngramsModel.add_corpus(text, f"{participantId}_{questionIdx}")

def generateGPTManglingPrompt():
	if len(responses.keys()) == 0:
		return "Make me anew. Dig for something until you forgot what you were looking for."
	# Use ChatGPT on the "instructions" question, the third one
	participantResponses = []
	while len(participantResponses) < 3:
		participantIdx = math.floor(random() * len(responses.keys()))
		participantId = [k for k in responses.keys()][participantIdx]
		participantResponses = responses[participantId]
	return responses[participantId][2]

def generateMangled():
	global responses

	if len(responses.keys()) == 0:
		return mangleWithGPT("Make me anew. Dig for something until you forgot what you were looking for.")

	sel = random()
	if (sel < 0.5):
		# Use ngrams
		return ngramsModel.generate(stop_mode=ngrams.StopMode.MAX_ATOMS, n_atoms=50)
	else:
		# Use ChatGPT on the "instructions" question, the third one
		participantResponses = []
		while len(participantResponses) < 3:
			participantIdx = math.floor(random() * len(responses.keys()))
			participantId = [k for k in responses.keys()][participantIdx]
			participantResponses = responses[participantId]
		return mangleWithGPT(responses[participantId][2])

# Annotation Types

def determineAnnotationType(participantId):
	global responses
	participantResponses = responses[participantId]
	# TODO: Parse + choose annotation type
	# TODO: But first, also come up with the annotation types...
	pass

# -- ROUTES -- #

@app.route('/')
def index():
	return render_template('client.html')

@app.route('/api/beginSession', methods=["POST"])
def beginSession():
	return f"{{ \"participantId\": \"{uuid4()}\", \"numQuestions\": {len(QUESTIONS)}, \"questionText\": \"{QUESTIONS[0]}\"}}"

@app.route('/api/advanceQuestion', methods=["POST"])
def advanceQuestion():
	responseText = request.json['response']
	participantId = request.json['participantId']
	prevQuestionIdx = request.json['idx']

	storeResponse(participantId, responseText)
	addToNGrams(participantId, prevQuestionIdx, responseText)
	saveData()
	
	if prevQuestionIdx == len(QUESTIONS) - 1:
		# Advance to close page
		return f"{{ \"annotationType\": \"{determineAnnotationType(participantId)}\" }}"
	else:
		return f"{{ \"questionText\": \"{QUESTIONS[prevQuestionIdx + 1]}\" }}"
	
@app.route('/api/printParticipant', methods=['POST'])
def printParticipant():
	participantId = request.json['participantId']
	# Print everything from this participant
	for i in range(len(QUESTIONS)):
		currText = responses[participantId][i]
		# print("Sending text:", currText)
		# sendToPrinter(currText, 1)
		# result = ser.readline()
		# print(result)
		success = False
		while not success:
			success = sendToPrinterWithHandshake(currText, 1)
	# print("Sending linebreak")
	# sendToPrinter("", 1)
	# result = ser.readline()
	# print(result)
	success = False
	while not success:
		success = sendToPrinterWithHandshake("", 1)
	ser.close()
	return "Success"

@app.route('/api/generateMangled')
def generateMangledEndpoint():
	return generateMangled()

@app.route('/api/testPrint0')
def routeSendToPrinter0():
	message = "Hello to printer 0 at time " + str(time.time())
	sendToPrinter(message, 0)
	return "Sent the following: \"" + message + "\""

@app.route('/api/testPrint1')
def routeSendToPrinter1():
	message = "Hello to printer 1 at time " + str(time.time())
	sendToPrinter(message, 1)
	return "Sent the following: \"" + message + "\""

@app.route('/api/printGPT')
def printRandomGPT():
	for i in range(200):
		possiblePrompts = [
			"can you give me a short answer for something creative that you heard of recently?",
			"in 10 words or less, can you give me an abstract process for creativity?",
			f"Imagine that you are a surrealist creative writer. Someone provides you the following instructions for how to write a very short, surreal, fragmented, choppy, ambiguous piece of provocative prose: \"{generateGPTManglingPrompt()}\" What do you write?"
		]
		idx = math.floor(random() * len(possiblePrompts))
		prompt = possiblePrompts[idx]
		completion = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[
				{"role": "user", "content": prompt}
			]
		)
		success = False
		while not success:
			success = sendToPrinterWithHandshake(completion.choices[0].message.content, 1, 5)
	return "Printed"


if USE_PREEXISTING_FILE is not None:
	print(f"Retrieving previously logged JSON data from \"{filename}\"...", file=sys.stderr)
	readData()
	# Load into ngrams
	for key, value in responses.items():
		for i, val in enumerate(value):
			# REFACTOR
			# Possibly organize just by response number instead of person
			# So you can manage all weights accordingly? But idk.
			ngramsModel.add_corpus(val, f"{key}_{i}")
else:
	print(f"Creating new JSON file at \"{filename}\"...", file=sys.stderr)
	saveData()

print("Connecting to serial...")
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD_RATE)

if __name__ == '__main__':
	print("Starting Flask server...\n", file=sys.stderr)
	app.run(debug = True)
