import numpy as np
import multiprocessing
import time
import os.path
from joblib import Parallel, delayed

def pl_fc_entails(symbols: list, KB_clauses: list, known_symbols: list):
	count = [0 for _ in range(len(KB_clauses))]
	symbols_dic = {}
	symbols_list = []
	known_symbols_list = []
	for symbol in symbols:
		symbols_dic[symbol[0] + ' ' + symbol[1] + ' ' + symbol[2]] = symbol
		symbols_list += [symbol[0] + ' ' + symbol[1] + ' ' + symbol[2]]
	for symbol in known_symbols:
		known_symbols_list += [symbol[0] + ' ' + symbol[1] + ' ' + symbol[2]]
	for i, c in enumerate(KB_clauses):
		count[i] = len(c.body)
	inferred = {s: False for s in symbols_list}
	agenda = known_symbols_list
	while len(agenda) != 0:
		p = agenda.pop(0)
		if not inferred[p]:
			inferred[p] = True
			for i, c in enumerate(KB_clauses):
				if p in c.body:
					count[i] -= 1
					if count[i] == 0:
						agenda += [c.conclusion]
	return inferred


class definite_clause:
	"""
        Attributes
        --------------

            body: a list of symbol(s) (Have to be integers), "body" of a definite clause
            conclusion: a single integer symbol, also called head
    """

	def __init__(self, body: list = [], conclusion: object = None):
		self.body = body
		self.conclusion = conclusion

	def set_body(self, body: list):
		self.body = body

	def set_conclusion(self, conclusion: int):
		self.conclusion = conclusion


class KB:
	def __init__(self, shape):
		self.KB_matrix = np.zeros(shape)


class Query:
	def __init__(self, relation, arg1, arg2):
		self.relation = relation
		self.arg1 = arg1
		self.arg2 = arg2


class Rule:
	def __init__(self, head, tails):
		self.head = head
		self.tails = tails
		self.arg_dict = {head[1]: [], head[2]: []}
		for i in range(len(tails)):
			self.arg_dict[tails[i][1]] = list(entities.keys())
			self.arg_dict[tails[i][2]] = list(entities.keys())
		self.current_pointer = [0 for _ in range(len(self.arg_dict))]

	def increment_counter(self, bit):
		self.current_pointer[bit] += 1
		i = bit
		while self.current_pointer[i] == len(list(self.arg_dict.values())[i]):
			self.current_pointer[i] = 0
			if i == len(self.current_pointer) - 1:
				self.current_pointer = [0 for _ in range(len(self.arg_dict))]
				return False
			self.current_pointer[i + 1] += 1
			i += 1
		return True

	def clear_counter(self):
		self.current_pointer = [0 for _ in range(len(self.arg_dict))]

	def evaluate_rule(self,arg_list):
		for tail in self.tails:
			relation=tail[0]
			e1=arg_list[tail[1]]
			e2=arg_list[tail[2]]
			if facts.KB_matrix[relations[relation],entities[e1],entities[e2]]==0:
				return False
		return True


def generate_definite_clause(rule):
	keys = list(rule.arg_dict.keys())
	atoms = {}
	clause_list = []
	while(True):
		body = []
		for i, key in enumerate(keys):
			atoms[key] = rule.arg_dict[key][rule.current_pointer[i]]
		if rule.evaluate_rule(atoms):
			for tail in rule.tails:
				relation = tail[0]
				e1 = atoms[tail[1]]
				e2 = atoms[tail[2]]
				body+=[relation+' '+e1+' '+e2]
			clause_list+=[definite_clause(body,rule.head[0]+' '+atoms[rule.head[1]]+' '+atoms[rule.head[2]])]
		if not rule.increment_counter(0):
			break
	rule.clear_counter()
	return clause_list


def string_to_rule(element):
	relation = element.split('(')[0]
	temp = element.split('(')[1]
	arg1 = temp[0]
	arg2 = temp[2]
	return [relation, arg1, arg2]


def check_in_facts(query):
	return facts.KB_matrix[relations[query[0]], entities[query[1]], entities[query[2]]] == 1

exp_category="Divisibility"
exp_name="1000"
start=time.time()
rule_path = os.path.dirname(__file__) + '/../'+exp_category+'/'+exp_name+'/rules.txt'
relation_path = os.path.dirname(__file__) + '/../'+exp_category+'/'+exp_name+'/'+exp_name+'data/relations.txt'
entity_path = os.path.dirname(__file__) + '/../'+exp_category+'/'+exp_name+'/'+exp_name+'data/entities.txt'
fact_path = os.path.dirname(__file__) + '/../'+exp_category+'/'+exp_name+'/'+exp_name+'data/facts.txt'
rules = []
entities = {}
relations = {}
count = 0

with open(entity_path) as file:
	lines = file.readlines()
	for line in lines:
		entities[line[:-1]] = count
		count += 1
count = 0
file.close()
with open(relation_path) as file:
	lines = file.readlines()
	for line in lines:
		relations[line[:-1]] = count
		count += 1
	for line in lines:
		relations['inv_' + line[:-1]] = count
		count += 1
file.close()
relations['equal'] = count
with open(rule_path) as file:
	lines = file.readlines()
	for line in lines:
		words = line.split(' ')
		temp = words[1].split('\t')
		words = [words[0]] + temp + words[2:]
		conf = float(words[0])
		head = words[2] + words[3]
		head = string_to_rule(head)
		tail = []
		for i in range(5, len(words), 2):
			tail += [words[i] + words[i + 1]]
			tail[-1] = tail[-1][:-1]
			tail[-1] = string_to_rule(tail[-1])
		rules += [Rule(head, tail)]
file.close()
facts = KB((len(relations) * 2, len(entities), len(entities)))
with open(fact_path) as file:
	lines = file.readlines()
	for line in lines:
		words = line.split('\t')
		facts.KB_matrix[relations[words[1]], entities[words[0]], entities[words[2][:-1]]] = 1
file.close()
clause_list = []
parrallel=False
if not parrallel:	
	for rule in rules:
		clause_list+=generate_definite_clause(rule)
else:
	num_cores = multiprocessing.cpu_count()
	results = Parallel(n_jobs=num_cores,  require='sharedmem')(delayed(generate_definite_clause)(rule) for rule in rules)
	for i in results:
		clause_list+=i	
symbols_list = []
known_symbols = []
for r in relations:
	for e1 in entities:
		for e2 in entities:
			symbols_list += [[r, e1, e2]]
			if facts.KB_matrix[relations[r], entities[e1], entities[e2]] == 1:
				known_symbols += [[r, e1, e2]]
print("prepared")
inferred = pl_fc_entails(symbols_list, clause_list, known_symbols)
with open("all_infered_"+exp_category+"_"+exp_name+".txt","w+") as file:
	for key, value in inferred.items():
		if value:
			file.write(key+"\n")
file.close()
end=time.time()
print(end-start)