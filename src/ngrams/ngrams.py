from enum import Enum, auto
from unittest import registerResult
import numpy as np
import re
import copy

class NextMethod(Enum):
	NORMAL_LIKELY = auto()
	NORMAL_UNLIKELY = auto()
	RANDOM = auto()

class StopMode(Enum):
	RUN_OUT = auto()
	MAX_ATOMS = auto()
	FIND_ATOM = auto()
	FIND_ATOM_WITH_MIN_ATOMS = auto()

class FirstMode(Enum):
	ATOMS = auto()
	BAG_OF_ATOMS = auto()

# FIXME: Not sure if this maintains the same shape of distribution. Probably not. Fix?
def invert_probs(probs):
	# If there's only one item, it has probability 1, so it makes no sense to invert it
	if len(probs.values()) == 1:
		return probs
	new_total = sum([(1 - val) for val in probs.values()])
	return {key: (1 - val) / new_total for key, val in probs.items()}

def select_stochastic_key(selector, probs):
	'''
	For a dictionary whose values add up to 1 (representing probabilities), choose a key
	based on a randomly chosen value. Acts a continuous index into discrete, probabilistic bins.
	
	selector (float): A value in [0, 1] representing the random number chosen.
	probs {key: [0, 1]}: The dictionary of probabilities from whose keys to choose.
	'''
	# order = sorted(probs.items(), key=lambda item: item[1])
	total_prob = 0
	for key, prob in probs.items():
		total_prob += prob
		if selector <= total_prob:
			return key
	return list(probs.keys())[-1] # Should never hit, but just in case

class NGramEdge:
	def __init__(self, next_atom, corpus_weights):
		self.next_atom = next_atom
		# TODO: Possibly also tag word intensities or other vars?
		self.counts = dict()
		self.weights = corpus_weights

	def add_from_corpus(self, corpus_name):
		if corpus_name not in self.counts:
			self.counts[corpus_name] = 0
		self.counts[corpus_name] += 1

	def probability(self):
		# For now, this is returning an un-normalized weighted count
		return sum([self.weights[name] * self.counts[name] for name in self.counts.keys()])

class NGramNode:
	def __init__(self, atom, corpus_weights):
		self.atom = atom
		self.edges = dict()
		self.weights = corpus_weights

	def add_next(self, next_atom, corpus_name):
		if next_atom not in self.edges:
			self.edges[next_atom] = NGramEdge(next_atom, self.weights)
		edge = self.edges[next_atom]
		edge.add_from_corpus(corpus_name)
	
	def next(self, next_method):
		probs = self.probabilities()
		if len(probs) == 0:
			return None
		if next_method == NextMethod.RANDOM:
			return np.random.choice(list(self.edges.values())).next_atom
		total = sum(probs.values()) # This is more so counts, so I'm normalizing here
		probs = {key: val / total for key, val in probs.items()}
		if next_method == NextMethod.NORMAL_UNLIKELY:
			probs = invert_probs(probs)
		# FIXME: This relies on proportional area, but it's probably not an orthodox normal distribution.
		# In fact, it's uniform. So that should likely be changed depending on how strictly I want
		# something to match the most/least likely option.
		selector = np.random.uniform(0, 1)
		return select_stochastic_key(selector, probs)

	def probabilities(self):
		return {atom: edge.probability() for atom, edge in self.edges.items()}

class NGramModel:
	def __init__(self, atomize, recombine_atoms, n=1, n_order_key_separator="|", key_preprocessor=lambda x: x):
		self.atomize = atomize
		self.recombine_atoms = recombine_atoms
		self.n = n
		self.n_order_key_separator = n_order_key_separator
		self.key_preprocessor = key_preprocessor
		# Define graph structure as dictionary with weight children
		self.nodes = dict()
		self.corpus_weights = dict()

	def concat_atoms_for_key(self, atoms):
		if not isinstance(atoms, list):
			return self.key_preprocessor(atoms)
		if len(atoms) == 1:
			# Preserve data type of singular atom
			return self.key_preprocessor(atoms[0])
		return self.n_order_key_separator.join([str(self.key_preprocessor(x)) for x in atoms])

	def add_transition(self, curr_atom, next_atom, corpus_name="default"):
		key = self.add_empty(curr_atom, corpus_name)
		self.nodes[key].add_next(next_atom, corpus_name)
		return key

	def add_empty(self, curr_atom, corpus_name="default"):
		key = self.concat_atoms_for_key(curr_atom)
		if key not in self.nodes:
			self.nodes[key] = NGramNode(curr_atom, self.corpus_weights)
		return key

	def add_corpus(self, text, corpus_name="default"):
		if corpus_name not in self.corpus_weights:
			self.corpus_weights[corpus_name] = 1
		# Split the text into tokens/atoms
		atoms = self.atomize(text)
		# Iterate over atoms and keep a rolling list of size n to serve as the key
		# Add transitions for that key to the next atom (current index)
		for i in range(self.n, len(atoms)):
			curr_atoms = atoms[i - self.n : i]
			next_atom = atoms[i]
			self.add_transition(curr_atoms, next_atom, corpus_name)
		self.add_empty(atoms[len(atoms) - self.n : len(atoms) + 1])

	def add_corpus_from_file(self, filepath, corpus_name="default"):
		file = open(filepath, "r")
		text = "".join(file.readlines())
		self.add_corpus(text, corpus_name)

	def set_corpus_weight(self, corpus_name, weight):
		# Weights get passed down to all nodes and their edges as references
		if corpus_name not in self.corpus_weights:
			return
		self.corpus_weights[corpus_name] = weight

	def generate(self, first=None, first_mode=FirstMode.ATOMS, stop_mode=StopMode.RUN_OUT, n_atoms=None, stop_atom=None, next_method=NextMethod.NORMAL_LIKELY, first_search_preprocessor=lambda x: x):
		"""
		Performs a (random) walk across the corpus(es) to produce new text based on the source material.

		first (any, same as atom | any[] | None): The first atom(s) to use in the sequence to prime how the other atoms are found. If None, or if the desired value is not found, a first atom is chosen randomly.
		first_mode (FirstMode): How to use the first value. If ATOMS, then treats first as either a string or list of atoms to find sequentially. If BAG_OF_ATOMS, then treats first as a list of either strings or lists from which to choose and then find sequentially.
		stop_mode (StopMode): How to stop generation. If RUN_OUT, continues until there are no more edges, but may anot terminate. If MAX_ATOMS, then stops at or around n_atoms atoms. If FIND_ATOM, stops upon finding stop_atom. If FIND_ATOM_WITH_MIN_ATOMS, stops upon finding stop_atom only if has already gathered n_atoms.
		n_atoms (int): Used by StopMode.MAX_ATOMS and StopMode.FIND_ATOM_WITH_MIN_ATOMS; see stop_mode for more info.
		stop_atom (any, same as atom | list): Used by StopMode.FIND_ATOM and StopMode.FIND_ATOMS_WITH_MIN_ATOMS; see stop_mode for more info.
		next_method (NextMethod): Determines how to choose the next atom.
		first_search_preprocessor (any (atom) -> any (atom)): Determines how to preprocess keys in order to search for the desired first atom.
		"""
		if stop_mode == StopMode.MAX_ATOMS or stop_mode == StopMode.FIND_ATOM_WITH_MIN_ATOMS:
			assert n_atoms is not None, f"For stop mode {stop_mode}, n_atoms must be specified."
		if stop_mode == StopMode.FIND_ATOM or stop_mode == StopMode.FIND_ATOM_WITH_MIN_ATOMS:
			assert stop_atom is not None, f"For stop mode {stop_mode}, stop_atom (the atom on which to stop generating) must be specified."
		
		if stop_atom is not None and not isinstance(stop_atom, list):
			# FIXME: For the sake of scalability, change this because it assumes that atoms cannot be lists. That's a flawed assumption.
			stop_atom = [stop_atom]

		text = []

		if first_mode == FirstMode.BAG_OF_ATOMS and first is not None and isinstance(first, list):
			# First choose one of the atoms to use to prime the sequence, and then continue normally
			first = np.random.choice(first)

		# TODO: Some of these edge cases probably need to be better tested, but the elif -> if I've tested fairly thoroughly.
		# Choose first word to prime the sequence
		# If it's a list, just grab the node from the concatenated key
		if isinstance(first, list):
			if self.n > 1:
				assert len(first) == self.n, "Cannot yet use lists of length not equal to n as first for order-n models." # TODO: There's got to be a nice clean way to traverse stuff, just like in generate, and use that for list-based first.
				first = self.concat_atoms_for_key(first)
				found_key = False
				for key in self.nodes.keys():
					if first == first_search_preprocessor(key):
						found_key = True
						first = key
						break
				if not found_key:
					# Couldn't find the desired key, so default to choosing randomly
					first = None
			else:
				# If not higher order, try to find the list of atoms by traversing the model
				# TODO: Implement this!
				raise NotImplementedError("Cannot yet use lists as first for order-1 models.")
		elif first is not None and first not in self.nodes:
			# If it's not a list but the model is higher-order, then find lists that start with that atom
			if self.n > 1:
				# Find all keys that start with that atom
				keys = []
				for key in self.nodes.keys():
					try:
						if first_search_preprocessor(key).index(first + self.n_order_key_separator) == 0:
							keys.append(key)
					except:
						continue
				# If any were found, randomly choose between them
				if len(keys) > 0:
					first = np.random.choice(keys)
				else:
					# Couldn't find the desired key, so default to choosing randomly
					first = None
			# If it's not a list and the model is first order, ensure that first is a viable key (not just something that matches preprocessor)
			else:
				possible_keys = []
				for key in self.nodes.keys():
					if first_search_preprocessor(key) == first:
						possible_keys.append(key)
				if len(possible_keys) == 0:
					first = None
				else:
					first = np.random.choice(possible_keys)

		# If was given none, or finding first did not succeed, just choose a random key to start with
		if first is None:
			first = np.random.choice(list(self.nodes.keys()))

		node = self.nodes[first]
		text.extend(node.atom)

		# Keep on finding subsequent items
		while True:
			if stop_mode == StopMode.MAX_ATOMS and len(text) >= n_atoms:
				break
			result = node.next(next_method)
			if result is None:
				break
			key = self.concat_atoms_for_key(text[len(text) - self.n + 1 : len(text) + 1] + [result])
			if isinstance(self.nodes[key].atom, list):
				text.append(self.nodes[key].atom[-1]) # FIXME: Possibly wrong?
			else:
				text.append(self.nodes[key].atom)
			if stop_mode == StopMode.FIND_ATOM and result in stop_atom:
				break
			if stop_mode == StopMode.FIND_ATOM_WITH_MIN_ATOMS and result in stop_atom and len(text) >= n_atoms:
				break
			node = self.nodes[key]
		return self.recombine_atoms(text)

class NGramParams:
	def __init__(self, weights:dict=None, first=None, first_mode=FirstMode.ATOMS, stop_mode=StopMode.RUN_OUT, n_atoms=None, stop_atom=None, next_method=NextMethod.NORMAL_LIKELY, first_search_preprocessor=lambda x: x, n_runs=1, join_runs=lambda x: "\n".join(x)):
		self.weights = weights
		self.first = first
		self.first_mode = first_mode
		self.stop_mode = stop_mode
		self.n_atoms = n_atoms
		self.stop_atom = stop_atom
		self.next_method = next_method
		self.first_search_preprocessor = first_search_preprocessor
		self.n_runs = n_runs
		self.join_runs = join_runs
	
	def generate(self, model: NGramModel):
		# Set corpus weights
		if self.weights is not None:
			for corpus_name, weight in self.weights.items():
				model.set_corpus_weight(corpus_name, weight)
		# Nothing else to modulate, just generate using the given parameters
		results = []
		for i in range(self.n_runs):
			results.append(model.generate(self.first, self.first_mode, self.stop_mode, self.n_atoms, self.stop_atom, self.next_method, self.first_search_preprocessor))
		return self.join_runs(results)

	def extend(self, **kwargs):
		clone = copy.deepcopy(self)
		for key, value in kwargs.items():
			if hasattr(clone, key):
				setattr(clone, key, value)
		return clone

# Time signatures are probably a no... more excited about vaguely polyrhythmic.
# A little off. There but not there. Imperfect repetition. Human component of mistake.
# Can start with something with time signature and then mess it up if possible.
# The beat is samples of wind recordings from archives. Imagine some parts of the wind
# sound more like bass drums, some sound more like high hats, some characteristics.
# That's something unknown, and that's the improvisation between the machine and the wind.

def clean_unicode(x):
	replace = ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2015', '\u2E3A', '\u2E3B', '\uFE58', '\uFE63', '\uFF0D']
	hyphen_minus = '\u002D'
	for c in replace:
		x = x.replace(c, 2*hyphen_minus)
	transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”",  u"'''\"\"")])
	x = x.translate(transl_table)
	x = x.replace('\u2026', '...')
	return x

def word_symbol_atomize(x):
	cleaned = clean_unicode(x)
	p = re.compile("""([A-z]+|[!"#\$%&'()\*+,-./:;<=>?@\[\]^_`~])""")
	return p.findall(cleaned)

def word_symbol_space_atomize(x):
	cleaned = clean_unicode(x)
	# p = re.compile("""([A-z]+|[!"#\$%&'()\*+,-./:;<=>?@\[\]^_`~]|\s+)""")
	p = re.compile("""([A-zÀ-ÿ]+|\d+|[!"#\$%&'()\*+,-./:;<=>?@\[\]^_`~]|\s+)""")
	return p.findall(cleaned)

def clean_output_text(x: str):
	x = re.sub(r"\s*--\s*", "--", x)
	leftParenIdxs = []
	rightParensToRemoveIdxs = []
	for i in range(len(x)):
		if x[i] == "(":
			leftParenIdxs.append(i)
		elif x[i] == ")":
			if len(leftParenIdxs) == 0:
				rightParensToRemoveIdxs.append(i)
			else:
				leftParenIdxs.pop()
	for i in reversed(sorted(leftParenIdxs + rightParensToRemoveIdxs)):
		x = x[:i] + x[i+1:]
	quoteIdxs = list(re.finditer('"', x))
	if len(quoteIdxs) == 1:
		i = quoteIdxs[0].span()[0]
		x = x[:i] + x[i+1:]

	#processing of spanish characters
	x = x.replace("ñ", "A4")
	x = x.replace("á", "A0")
	x = x.replace("é", "82")
	x = x.replace("í", "A1")
	x = x.replace("ó", "A2")
	x = x.replace("ú", "A3")
	x=new_lines(x)
	return x

def new_lines(s: str):
	to_trim = s 
	result = ""
	while len(to_trim)>30:
		i = to_trim.rindex(" ", 0, 27)
		result = result+to_trim[:i]+"\n"
		to_trim = to_trim[i+1:]
	result+=to_trim
	return result
def create_test_models():
	def join_runs(results):
		text = ""
		for i, result in enumerate(results):
			text += f"{i+1}. {result[0].upper()}{result[1:]}"
			if i < len(results) - 1:
				text += "\n\n"
		return text

	FIRST_BAG = ["do", "step", "try", "sing", "resonate", "vibrate", "perform", "sample", "experience", "why", "who", "what", "where", "how", "look", "listen", "pay", "think", "dance"]
	BASE_PARAMS = NGramParams(
		first=FIRST_BAG,
		first_mode=FirstMode.BAG_OF_ATOMS,
		stop_mode=StopMode.FIND_ATOM,
		stop_atom=[".", "?", "!"],
		first_search_preprocessor=lambda x: x.lower(),
		n_runs=3,
		join_runs=join_runs
	)
	
	models = []

	for i in range(1, 11):
		model = NGramModel(word_symbol_space_atomize, lambda x: "".join(x), n=i)
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/nicoleexamtext.txt", "exam")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/knittingtext.txt", "knitting")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/anzalduaborderlands.txt", "anzaldua")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/mistralpoemas.txt", "mistral")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/oliverosdeeplisteningcommentary.txt", "oliveroscommentary")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/oliverosdeeplisteningexercises.txt", "oliverosexercises")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/oliverostranslation.txt", "oliverostranslation")
		model.add_corpus_from_file("/Users/membranes/Documents/corpuses/spanishtext.txt", "spanish")
		models.append(model)

	def run(n):
		print(BASE_PARAMS.generate(models[n - 1]))
	
	return models, BASE_PARAMS, run

if __name__ == '__main__':
	# atomize = lambda x: x.split(" ")
	# atomize = lambda y: list(filter(lambda x: x != "", re.split(r"\s", y)))

	# model = NGramModel(word_symbol_atomize, lambda atoms: " ".join(atoms), n=2)
	# model.add_corpus("hello world i am a robot speaking to you from the void this is my language these are my words i can write in very reasonable english sentences i should maybe reuse some words because i am writing and this is what it is to write because the quick brown fox jumped over the lazy dog and that is a sentence and these are many sentences so i wonder what the model will do with this text because it really is interesting or maybe it is not but we will see now notice that i am not using any fancy contractions or anything like that just basic words in english and we will observe the results of this very short and simple test to see what connections are forged ok bye")
	# print(model.nodes['to'].next(NextMethod.NORMAL_UNLIKELY))
	# TODO: Test generation with n = 1 and n > 1

	# TODO: Test bag of atoms with bag of imperative verbs (and some question words)
	# TODO: Create corpuses, try weighting them, etc.
	# TODO: Create 1 hour loop (or shorter for testing) to produce rituals automatically
	# TODO: Test modulating using the wind

	# model = NGramModel(word_symbol_space_atomize, lambda atoms: "".join(atoms), n=2)
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/nicoleexamtext.txt", "exam")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/knittingtext.txt", "knitting")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/anzalduaborderlands.txt", "anzaldua")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/mistralpoemas.txt", "mistral")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/oliverosdeeplisteningcommentary.txt", "oliveroscommentary")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/oliverosdeeplisteningexamples.txt", "oliverosexamples")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/oliverostranslation.txt", "oliverostranslation")
	# model.add_corpus_from_file("/Users/membranes/Documents/corpuses/spanishtext.txt", "spanish")
	# print(model.generate(stop_mode=StopMode.FIND_ATOM, stop_atom=".", first="do", first_search_preprocessor=lambda x: x.lower()))
	models, BASE_PARAMS = create_test_models()
	while True:
		BASE_PARAMS.generate(models[0])
