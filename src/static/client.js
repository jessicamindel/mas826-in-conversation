console.log("hit, client js loaded");

const els = {
	introContainer: document.querySelector('#intro-container'),
	qaContainer: document.querySelector('#qa-container'),
	resultContainer: document.querySelector('#result-container'),
	question: document.querySelector('#question'),
	response: document.querySelector('#response'),
	nextButton: document.querySelector('#next-button'),
	beginButton: document.querySelector('#begin-button'),
};

var participantId = -1;
var questionIdx = 0;
var numQuestions = -1;

function resetDisplay() {
	els.qaContainer.style.opacity = 0;
	els.resultContainer.style.opacity = 0;
	els.response.value = "";
	setTimeout(() => {
		els.qaContainer.style.display = 'none';
		els.resultContainer.style.display = 'none';

		els.introContainer.style.opacity = 0;
		els.introContainer.style.display = 'block';
		setTimeout(() => {
			els.introContainer.style.opacity = 1;
		}, 100);
	}, 500);
}

function startQuestions() {
	els.introContainer.style.opacity = 0;
	els.resultContainer.style.opacity = 0;
	setTimeout(() => {
		els.introContainer.style.display = "none";
		els.resultContainer.style.display = "none";

		els.qaContainer.style.opacity = 0;
		els.qaContainer.style.display = "block";
		setTimeout(() => {
			els.qaContainer.style.opacity = 1;
		}, 100);
	}, 500);

	// Start a session with a unique id
	// Source: https://gomakethings.com/how-to-use-the-fetch-api-with-vanilla-js/
	fetch('/api/beginSession', {
		method: 'POST'
	}).then(function (response) {
		// The API call was successful!
		if (response.ok) {
			return response.json();
		} else {
			return Promise.reject(response);
		}
	}).then(function (data) {
		// This is the JSON from our response
		console.log(data);
		participantId = data.participantId;
		questionIdx = 0;
		numQuestions = data.numQuestions;
		els.question.innerText = data.questionText;
	}).catch(function (err) {
		// There was an error
		console.warn('Something went wrong.', err);
	});
}

function submitQuestionAndAdvance() {
	let response = els.response.value;

	if (questionIdx >= numQuestions) return;

	fetch('/api/advanceQuestion', {
		method: 'POST',
		"headers": {"Content-Type": "application/json"},
		"body": JSON.stringify({
			participantId: participantId,
			response: response,
			idx: questionIdx,
		}),
	}).then(res => {
		if (res.ok) {
			return res.json();
		} else {
			return Promise.reject(res);
		}
	}).then(data => {
		console.log(data);
		// TODO: Add nice fade in/out animations

		questionIdx++;
		if (questionIdx >= numQuestions) {
			// Advance to final page with annotation type
			els.qaContainer.style.opacity = 0;
			els.introContainer.style.opacity = 0;
			setTimeout(() => {
				els.qaContainer.style.display = 'none';
				els.introContainer.style.display = 'none';

				els.resultContainer.style.opacity = 0;
				els.resultContainer.style.display = 'block';
				setTimeout(() => {
					els.resultContainer.style.opacity = 1;
				}, 100);
			}, 500);
			// TODO: Pipe in new input for this
			// setTimeout(() => {
			// 	resetDisplay();
			// }, 20 * 1000);

			// In the background, print the output
			fetch('/api/printParticipant', {
				method: 'POST',
				"headers": {"Content-Type": "application/json"},
				"body": JSON.stringify({
					participantId: participantId,
				}),
			}); // We don't care about the result, let it happen in the background
		} else {
			els.response.value = '';
			els.question.innerText = data.questionText;
		}
	}).catch(err => {
		console.warn('Something went wrong.', err);
	});
}

els.beginButton.addEventListener("click", () => {
	startQuestions();
});

els.nextButton.addEventListener("click", () => {
	submitQuestionAndAdvance();
});

resetDisplay();
