#!/usr/bin/env python3

"""
Script adapted from https://github.com/techiaith/welsh-lts/blob/master/llef.py

Hawlfraint (c) 2014, Prifysgol Bangor
Copyright (c) 2014, Bangor University

Rhoddir caniatâd, yn rhad ac am ddim, i unrhyw berson sydd yn cael copi o'r
meddalwedd hwn a ffeiliau dogfennaeth cysylltiedig (y "Meddalwedd"), i
ymdrin â'r Meddalwedd heb gyfyngiad, gan gynnwys heb gyfyngiad yr hawl i
ddefnyddio, copïo, addasu, cyfuno, cyhoeddi, dosbarthu, is-drwyddedu, ac/neu
gwerthu copïau o'r Meddalwedd, ac i ganiatáu i bersonau y rhoddir y
Meddalwedd iddynt i wneud hynny, yn amodol ar yr amodau canlynol:

Rhaid cynnwys y hysbysiad hawlfraint uchod a'r hysbysiad caniatâd hwn ym
mhob copi neu ran sylweddol o'r Meddalwedd.

DARPERIR Y MEDDALWEDD "FEL Y MAE", HEB WARANT O UNRHYW FATH, NAILL AI WEDI'I
FYNEGI NEU YMHLYG, GAN GYNNWYS OND HEB FOD YN GYFYNGEDIG I WARANTAU
MARSIANDWYAETH, FFITRWYDD AT BWRPAS ARBENNIG A PHEIDIO Â THORRI AMODAU. NI
FYDD YR AWDURON NA'R DEILIAID HAWLFRAINT YN ATEBOL DAN UNRHYW AMGYLCHIADAU
AM UNRHYW HAWLIAD, IAWNDAL, NEU ATEBOLRWYDD ARALL, BOED MEWN ACHOS CONTRACT,
CAMWEDD NEU FEL ARALL, SYDD YN DEILLIO O, ALLAN O NEU MEWN CYSYLLTIAD Â'R
MEDDALWEDD, EI DDEFNYDD, NEU DRAFODION ERAILL AG EF.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Prif awduron (yn nhrefn yr wyddor):
Principal authors (in alphabetical order):

David Chan <d.chan@bangor.ac.uk>
Sarah Cooper <s.cooper@bangor.ac.uk>
"""



import logging
import re
import sys
import traceback


_logger = logging.getLogger(__name__) 


def u8(ob):
    # Not needed in Python 3 as strings are Unicode by default
    return ob

class Syllable(object):
    def __init__(self, onset, vowels, coda, is_final):
        self.onset = onset
        self.vowels = vowels
        self.coda = coda
        self.is_final = is_final

    def __repr__(self):
        return 'Syllable(onset=%r, vowels=%r, coda=%r)' % (
            self.onset,
            ''.join(self.vowels) if self.vowels else self.vowels,
            ''.join(self.coda) if self.coda else '',
        )

    def reprShort(self):
        return '(%s %s %s%s)' % (
            (''.join(self.onset) if self.onset else '.'),
            (''.join(self.vowels) if self.vowels else '.'),
            (''.join(self.coda) if self.coda else '.'),
            ''  # Kill the dollars temporarily because it's messing up diff. ('$' if self.is_final else ''),
        )


ONSETS = tuple(tuple(x.split('-')) for x in """
    ch-d ll-n
    s-b-r s-b-l s-t-r s-g-l s-g-r
    p-l p-n p-r ff-l ff-r b-l b-n b-r mh-l mh-r f-l f-r m-l m-r
    t-l t-r th-l th-r d-l d-r nh-l nh-r dd-r n-r s-d s-ff
    c-l c-r c-n ch-l ch-r ch-n g-l g-n g-r ngh-l ngh-r ngh-n ng-l ng-r
    g-w c-w ch-w
    s-b s-t s-g s-l s-m s-n
    ng-wl ng-wn ng-wr
    g-wl g-wn g-wr
    wl wn wr
    p b t d c g m n ng r ff f th dd
    s sh ch ll w
    l ts chw dz h mh nh ngh rh z
    j
""".split())

CODAS = tuple(tuple(x.split('-')) for x in """
    n-c-r n-c-l n-c-s n-d-r n-t-r n-d-l m-b-l m-b-r l-ts l-d-s r-ts g-l-s
    s-t-r m-p-l s-t-l s-ts r-p-s ff-ts c-ts n-ts m-p-s
    r-ts n-d-s s-g-l r-f-s d-l-s r-d-s c-ts r-c-s n-c-t
    f-r c-r g-l th-r c-l n-s d-r p-r t-r d-n th-m
    s-l ff-r ff-l f-l b-r p-l g-r g-n t-l ch-r s-m
    n-t r-dd r-n r-ch r-th s-t s-g ll-t n-s l-ch c-s r-s n-c r-t r-f r-d
    l-s ts n-d m-p r-m r-c r-ff ng-l c-t ff-t l-d l-t p-s d-s l-c m-s l-m
    l-ff r-p l-f dd-f s-b r-ts l-p m-l r-l n-ts t-t n-th ng-s m-b ff-s
    f-s p-t r-ll g-s r-g l-ts ch-t th-s l-b r-b b-s n-j s-c ff-ts
    m-ff m-n l-g th-t s-ts
    w-n
    n-n r-r f-n b-l d-l d-r
    tsh
    p b t d c g m n ng r ff f th dd
    s sh ch ll w
    l ts dz rh z
    j
""".split())

STRESSED_EXCEPTIONS = {
	'a': ('A'),
	'ag': ('A', 'g'),
	'â': ('A'),
	'ei': ('EI'),
	'i': ('i'),
	'o': ('o'),
	'dy': ('d', '@',),
	'fy': ('f', '@',),
	'mi': ('m', 'I',),
	'y': ('@',),
	'ych': ('@', 'ch',),
	'yng': ('@', 'ng',),
	'yn': ('@', 'n',),
	'ym': ('@', 'm',),
	'yr': ('@', 'r',),
	'cilometr': ('c', 'I', 'l', 'O1', 'm', 'E', 't', '@', 'r'),
	'chilometr': ('ch', 'I', 'l', 'O1', 'm', 'E', 't', '@', 'r'),
	'gilometr': ('g', 'I', 'l', 'O1', 'm', 'E', 't', '@', 'r'),
	'hwyr': ('h', 'WY1', 'r'),
	'llwyr': ('ll', 'WY1', 'r'),
	'lwyr': ('l', 'WY1', 'r'),
	'metr': ('m', 'E1', 't', '@', 'r'),
	'fetr': ('m', 'E1', 't', '@', 'r'),
	'litr': ('l', 'I1', 't', '@', 'r'),
	'ochr': ('O1', 'ch', '@', 'r'),
	'theatr': ('th', 'E1', 'A', 't', '@', 'r'),
	'trwy': ('t', 'r', 'WY1'),
	'thrwy': ('th', 'r', 'WY1'),
	'drwy': ('d', 'r', 'WY1'),
	'trwy’r': ('t', 'r', 'WY1', 'r'),
	'thrwy’r': ('th', 'r', 'WY1', 'r'),
	'drwy’r': ('d', 'r', 'WY1', 'r'),
	'dirprwy': ('d', 'I1', 'r', 'p', 'r', 'WY'),
	'ddirprwy': ('dd', 'I1', 'r', 'p', 'r', 'WY'),
	'dirprwy’r': ('d', 'I1', 'r', 'p', 'r', 'WY', 'r'),
	'ddirprwy’r': ('dd', 'I1', 'r', 'p', 'r', 'WY', 'r'),
	'dsilis': ('j', 'I1', 'l', 'I', 's'),
	'ïodin': ('I1', 'O', 'd', 'I', 'n'),
}

def get_stressed_phones(word):
    if word in STRESSED_EXCEPTIONS:
        return tuple(STRESSED_EXCEPTIONS[word])
    phones = []
    syllables = get_syllables(word)
    if syllables is None:
        return []
    for s in syllables:
        phones.extend(s.onset or ())
        phones.extend(s.vowels)
        phones.extend(s.coda or ())
    return phones


def get_syllables(word):
	unstressed_phones, apostrophe_phone_indexes = get_unstressed_phones(word)
	syllables = split_syllables(unstressed_phones, apostrophe_phone_indexes)
	if syllables is not None:
		add_schwa(syllables)
		add_stress(syllables)
		remove_accents(syllables)
	return syllables


def add_stress(syllables):
	"""modifies in place"""
	# Stress at the (last) circumflex/acute accent, if there is one
	for syll in reversed(syllables):
		for i in reversed(range(len(syll.vowels))):
			v = syll.vowels[i]
			if re.match(r'[ÂÊÎÔÛŴŶÁÉÍÓÚẂÝ]', v):
				syll.vowels[i] = v + '1'
				return

	# Else stress the penult (or the last syllable for monosyllabic words)
	if len(syllables) >= 2:
		stressed = syllables[-2]
	else:
		stressed = syllables[0]
	stressed.vowels[-1] = stressed.vowels[-1] + '1'


ACCENT_REPLACEMENTS = dict(zip('ÂÊÎÔÛŴŶÁÉÍÓÚẂÝÀÈÌÒÙẀỲÄËÏÖÜẄŸ', 'AEIOUWYAEIOUWYAEIOUWYAEIOUWY'))

def remove_accents(syllables):
	"""modifies in place"""
	for syll in syllables:
		newVowels = []
		for v in syll.vowels:
			newVowels.append(''.join(ACCENT_REPLACEMENTS.get(ch, ch) for ch in v))
		syll.vowels = newVowels


SCHWA_REPLACEMENTS = {
	('Y',): ('@',), # usual case, including w-cons + Y, e.g. chwythu. (But there are complexities: gwynion etc)
	('I', 'Y'): ('I', '@'), # miliynau, cwestiynau
	('YW',): ('@W',), # amryw < cywion < cywrain. Oh dear. @ better than Y if in doubt
	# Don't do WY: gwyddel ayb.
}

def add_schwa(syllables):
	"""modifies in place"""
	for s in syllables:
		if not s.is_final:
			old = tuple(s.vowels)
			s.vowels = list(SCHWA_REPLACEMENTS.get(old, old))


def split_syllables(orig_phones, apostrophe_phone_indexes):
	"""Split into (C* V+ C*) groups.

	This algorithm is left-greedy, leaving consonants with the preceding vowel.
	That's what we want so that position in the stressed syllable can reflect
	vowel length.
	"""
	phones = orig_phones[:]
	syllables = []


	offset = 0

	while True:
		if not phones:
			break
		found_onset = None
		for onset in ONSETS:
			if tuple(phones[:len(onset)]) == onset:
				found_onset = onset
				phones = phones[len(onset):]
				break
		
		found_vowels = []
		while phones and is_vowel_phone(phones[0]):
			vowel = phones[0]
			found_vowels.append(vowel)
			phones = phones[1:]
			if vowel != 'I':
				# Allow 'iad' etc to be one syllable
				break
		if not found_vowels:
			raise ValueError('No vowels in syllable: orig_phones=%r, phones=%r' % (orig_phones, phones))

		found_coda = None
		for coda in CODAS:
			if tuple(phones[:len(coda)]) == coda:
				found_coda = coda
				phones = phones[len(coda):]
				break
		syllable_length = len(found_onset or ()) + len(found_vowels) + len(found_coda or ())
		precoda_length = len(found_onset or ()) + len(found_vowels)

		# is_final if end of word, or if there is an apostrophe after the nucleus
		is_final = (not phones)
		for i in range(offset + precoda_length + 1, offset + syllable_length + 1): # + 1 to look before the next syllable
			if i in apostrophe_phone_indexes:
				is_final = True
	
		syllables.append(Syllable(found_onset, found_vowels, found_coda, is_final))
	return tuple(syllables)


def extract_apostrophes(seq):
	inputSeq = list(seq)
	outputSeq = []
	apostropheIndexes = set()
	offset = 0
	for i in range(len(inputSeq)):
		elt = inputSeq[i]
		if elt == u"'":
			apostropheIndexes.add(i - offset)
			offset += 1
		else:
			outputSeq.append(elt)
	return outputSeq, apostropheIndexes


UNSTRESSED_EXCEPTIONS = {}
for line in """
moyn: m OE n
dy: d @
fy: f @
y: @
ych: @ ch
ym: @ m
yn: @ n
yr: @ r
Celsius: c E l s I W s
Gelsius: g E l s I W s
Chelsius: ch E l s I W s
Nghelsius: ngh E l s I W s
trapesiymau: t r A p E s I @ m AU
drapesiymau: d r A p E s I @ m AU
thrapesiymau: th r A p E s I @ m AU
nhrapesiymau: nh r A p E s I @ m AU
acasiâu: A c A s I Â U
hacasiâu: h A c A s I Â U
ffantasiâu: ff A n t A s I Â U
siwmper: sh W m p E r
siwmperi: sh W m p E r I
siwed: s IW E d
siw: s IW
siwio: s IW I o
siwper: s IW p E r
gwysiwr: g w Y s I W r
wysiwr: w Y s I W r
bwa: b W A
fwa: f W A
mwa: m W A
dwi: d W I
wnion: W n I O n
hwnion: h W n I O n
wnionyn: W n I O n Y n
hwnionyn: h W n I O n Y n
wnionod: W n I O n O d
hwnionod: h W n I O n O d
chwlomb: ch W l O m b
chwlombau: ch W l O m b AU
cwlomb: c W l O m b
cwlombau: c W l O m b AU
gwlomb: g W l O m b
gwlombau: g W l O m b AU
nghwlomb: ngh W l O m b
nghwlombau: ngh W l O m b AU
gwlwm: g W l W m
wlna: W l n A
hwlna: h W l n A
diwlychol: d I wl @ ch O l
ewlychol: E wl @ ch O l
hewlychol: h E wl @ ch O l
ewlychu: E wl @ ch U
hewlychu: h E wl @ ch U
diwnïad: d I wn Ï A d
ddiwnïad: dd I wn Ï A d
diwraidd: d I wr AI dd
ddiwraidd: dd I wr AI dd
diwreiddio: d I wr EI dd I O
ddiwreiddio: dd I wr EI dd I O
diwres: d I wr E s
ddiwres: dd I wr E s
dwywreiciaeth: d WY wr EI c I AE th
ddwywreiciaeth: dd WY wr EI c I AE th
dwywreigiaeth: d WY wr EI g I AE th
ddwywreigiaeth: dd WY wr EI g I AE th
dwywreigiol: d WY wr EI g I O l
ddwywreigiol: dd WY wr EI g I O l
Llandegwning: ll A n d E g W n I ng
Landegwning: l A n d E g W n I ng
emwladu: E m W l A d U
hemwladu: h E m W l A d U
dramaeiddio: d r A m A EI dd I O
Eirawen: EI r A w E n
ffawydd: ff A WY dd
ffawydden: ff A WY dd E n
bioymoleuedd: b I O @ m O l EU E dd
fioymoleuedd: f I O @ m O l EU E dd
mioymoleuedd: m I O @ m O l EU E dd
Gelliwastad: g E ll I w A s t A d
Ngelliwastad: ng E ll I w A s t A d
microeiliad: m I c r O EI l I A d
ficroeiliad: f I c r O EI l I A d
microeiliadau: m I c r O EI l I A d AU
ficroeiliadau: f I c r O EI l I A d AU
seroeiddio: s E r O EI dd I O
gloeuni: g l o EU n i
cilowat: c I l O w A t
gilowat: g I l O w A t
chilowat: ch I l O w A t
nghilowat: ngh I l O w A t
Niwrowyddorau: n IW r O w Y dd o r AU
ffocsls: ff O c s @ l s
hyrdls: h @ r d @ l s
fangls: f A n g @ l s
jyngls: j @ n g @ l s
mangls: m A n g @ l s
metr: m E t @ r
fetr: f E t @ r
cilometr: c I l O m E t @ r
chilometr: ch I l O m E t @ r
nghilometr: ngh I l O m E t @ r
gilometr: g I l O m E t @ r
litr: l I t @ r
theatr: th E A t @ r
ochr: O ch @ r
""".strip().split('\n'):
	word, phonestring = line.split(': ')
	UNSTRESSED_EXCEPTIONS[word] = phonestring.split()


# IW EW AW YW OW OI AI EI AU AE OE WY EU UW


#mwynhau Lasa    m WY nh AU1     m WY n h AU1
#yn haul Asia	    @ nh AU1 l      @ n h AU1 l

# Bantsaeson   b A n ts AE1 s O n      b A n t s AE s O n
# mats aeddfed  m A1 ts AE1 dd f E d    m A1 t s AE1 dd f E d

# yn Tsar	 @ n ts A1 r	   


class LogicError(RuntimeError):
	pass

def split_chars(word):
	return re.findall(r"ch|dd|ff|ngh|mh|nh|ng|ll|ph|rh|th|tsh|ts|sh|[bcdfghjlmnprst']|[aeouâêîôûäëïöüáéíóúàèìòùŵŷẅÿẃýẁỳ]|[iwy]", word, re.I|re.U)

def is_simple_cons(ch):
	return re.match(r'ch|dd|ff|ng|ll|ph|rh|th|mh|nh|ngh|tsh|ts|sh|[bcdfghjlmnprst]', ch, re.I|re.U)

def is_simple_vowel_cluster(ch):
	# all vowels except i/w/y as these may be consonantal
	return re.match(r'[aeouâêîôûŵŷäëïöüẅÿáéíóúẃýàèìòùẁỳ]+', ch, re.I)

def is_possible_vowel_cluster(ch):
	# all possible vowels, including i/w/y which may be consonantal
	return re.match(r'[aeiouwyâêîôûŵŷäëïöüẅÿáéíóúẃýàèìòùẁỳ]+', ch, re.I)

def is_vowel_phone(ff):
	return ff.isupper() or ff == '@'

# 1. Get phones, disambiguating /si/ from /s I/ and /W [lnr]/ from /w[lnr]/. Simple vowels (not iwy) are clustered.
# TODO: ts: rules or exceptions
# TODO: soft chw: exceptions (chwa vs chwe vs ymchwilio)
# 2. Find diphthongs
#  XXX [aeiou]wyd$ bob tro yn 'WY d'
#  XXX ^diw yn issue (ddim yn issue yng nghanol geiriau)
#  xxx wyw (obviously) yn issue. wy[aeiouy] yn biafio
#  xxx ywy (obviously) yn issue. yw[aeiouw] yn biafio
# 3. Break syllables, making syllable codas as big as is legal (because consonant length depends on previous vowel stress)
# 4. Add stress
# 5. Add final-syllableness.
# 5.1. Disambiguate Y from @.
# 6. Add vowel length.
# 7. Label vowels with stress


def get_unstressed_phones(word):
	phones = []
	apostrophePhoneIndexes = set()
	if word in UNSTRESSED_EXCEPTIONS:
		return UNSTRESSED_EXCEPTIONS[word]
	partsAndApostrophes = split_chars(word.lower())
	parts, apostrophePartIndexes = extract_apostrophes(partsAndApostrophes)

	dictparts = dict((i, parts[i]) for i in range(len(parts)))
	for i in range(len(parts)):
		if i in apostrophePartIndexes:
			apostrophePhoneIndexes.add(len(phones))
		append_phone(phones, parts, i, apostrophePartIndexes)
	phones = join_diphthongs(phones)
	return tuple(phones), apostrophePhoneIndexes


def append_phone(phones, parts, i, apostrophePartIndexes):
	pre = parts[i - 1] if i - 1 >= 0 else '^'
	now = parts[i]
	post = parts[i + 1] if i + 1 < len(parts) else '$'
	# tail and tailstr are unprocessed ouput from split_chars
	tail = tuple(parts[i + 1:])
	tailstr = ''.join(tail)

	# 0. Skip letters that were already included as part of a cluster
	if phones and phones[-1] == 'wl' and now == 'l':
		return
	elif phones and phones[-1] == 'wn' and now == 'n':
		return
	elif phones and phones[-1] == 'wr' and now == 'r':
		return
	# 1. Disambiguate vowels and consonants.
	# 1.1 Simple cases: unambiguous vowels / consonants
	elif now == 'ph':
		phones.append('ff')
	elif is_simple_cons(now):
		phones.append(now)
	elif is_simple_vowel_cluster(now) or now == 'y':
		phones.append(now.upper())
	# 1.2 The letter 'i' can be /I/ or part of /sh/
	elif now == 'i':
		append_i(phones, parts, i, pre, now, post, tail, tailstr)
	# 1.3 The letter 'w' can be /W/ or part of /wl/, /wn/, or /wr/
	elif now == 'w':
		append_w(phones, parts, i, pre, now, post, tail, tailstr, apostrophePartIndexes)
	else:
		#raise LogicError("This code should never be reached: " + repr((phones, parts, i, pre, now, post, tail)))
		print("append_phone: This code should never be reached: " + repr((phones, parts, i, pre, now, post, tail)), file=sys.stderr)

def append_i(phones, parts, i, pre, now, post, tail, tailstr):
	"""Handles 'i': determines whether part of 'si'=/sh/"""
	if pre != 's':
		phones.append('I')
	elif is_simple_cons(post):
		phones.append('I')
	elif post == 'i':
		# sI + ir|id|iff|it|ith
		phones.append('I')
	elif post == 'u':
		# Latin 'sius' etc
		phones.append('I')
	elif ''.join(phones).endswith('rAWs') and now == 'i':
		# trawsieithu, trawsiwerydd etc
		phones.append('I')
	elif post == 'y':
		phones.pop() # remove 's'
		phones.append('sh')
	elif is_simple_vowel_cluster(post) and post[0] != 'w':
		phones.pop() # remove 's'
		phones.append('sh')
	elif tailstr.startswith('wy'):
		# si + wyd|wyf
		# Pronunciation varies, so use 's I wy' and let triphones disambiguate
		phones.append('I')
	elif tailstr.startswith('wm'):
		# calsiwm etc
		phones.append('I')
	elif tailstr.startswith('wl'):
		# capsiwl, insiwleiddio etc
		# Pronunciation varies, so use 's IW l' and let triphones disambiguate
		phones.append('I')
	elif post == 'w':
		phones.pop() # remove 's'
		phones.append('sh')
	elif not tail:
		phones.append('I')
	else:
		#raise LogicError("This code should never be reached: " + repr((phones, parts, i, pre, now, post, tail)))
		print("append_i: This code should never be reached: " + repr((phones, parts, i, pre, now, post, tail)), file=sys.stderr)


def append_w(phones, parts, i, pre, now, post, tail, tailstr, apostrophePartIndexes):
	# 1.3.1 Handle simple cases (not w[lnr]<vowel>)
	if len(tail) < 2 or tail[0] not in 'lnr' or not is_possible_vowel_cluster(tail[1]):
		#phones.append('W')
		phones.append(get_type_of_w(phones, parts, i, pre, now, post, tail, tailstr, apostrophePartIndexes))
		return

	# 1.3.2 Handle w[lnr]<vowel>: may append /wl/, /wn/, or /wr/ .
	# If so, the l|n|r will be omitted in the next loop iteration.
	if is_possible_vowel_cluster(pre):
		phones.append('W')
	elif re.match(r'^(ryw|ria|ro)', tailstr):
		# gwryw, wriaeth|wriad|wrian, wrol|wron|wrogaeth
		phones.append('W')
	elif ''.join(phones[-2:]) == 'sg' and post == 'l':
		# *sgwl*
		phones.append('W')
	elif (pre == 'g' and ''.join(phones[-2:]) != 'sg') or pre == '^':
		if post == 'l':
			# XXX # except: gwlio, gwlyddyn, gwlaw
			# gwlydd g wl Y dd
			phones.append('wl')
		elif post == 'n':
			phones.append('wn')
		elif post == 'r':
			# XXX # except: gwrymiog gwrw
			# gwra gwrwst gwryd gwrhydau gwryf(-oedd)
			# gwrwst
			# gwrym x2 gwrymio gwrymiog gwrysg gwrysgen
			phones.append('wr')
	# Compounded mutated gw[lnr]- is /w[lnr]/ . Approximate this by looking
	# for likely compounds (mined from the hunspell word list: the tests are
	# 100% accurate for that list)
	elif post == 'l' and re.match(r'^(lad|ledydd|ledd|leidydd|latgar|orwlych)', tailstr):
		phones.append('wl')
	elif post == 'n' and (re.match(r'^(neud|neuthur)', tailstr) or (tuple(phones[-2:]) == ('y', 'm') and post == 'n')):
		phones.append('wn')
	elif post == 'r' and re.match(r'^(rand|rend|ragedd|raig|reidd)', tailstr):
		phones.append('wr')
	else: 
		phones.append('W')

def remove_inner_space(m):
	return re.sub(r'(?<=\S)\s+(?=\S)', '', m.group(1))

def join_diphthongs(phones):
	phonestring = ' '.join(phones)
	phonestring = re.sub(r'(I W|E W|A W|O W|Y W|O I|A I|E I|A U|A E|O E|W Y|E U|U W)', remove_inner_space, phonestring)
	return phonestring.split(' ')

def get_type_of_w(phones, parts, i, pre, now, post, tail, tailstr, apostrophePartIndexes):
	# lead and leadstr are unprocessed ouput from split_chars
	lead = tuple(parts[:i])
	leadstr = ''.join(lead)

	partstr = ''.join(parts)

	if re.match(r'^([bfm]wa|dwi)$', partstr, re.UNICODE):
		return 'W'
	elif tailstr.startswith('ryw'):
		return 'W'
	elif tailstr.startswith('y') and i + 2 in apostrophePartIndexes:
		return 'W' # wy'
	elif tailstr == 'yr':
		return 'w' # -wyr
	elif is_simple_cons(pre) and not tailstr.startswith('y'):
		return 'W'
	elif tailstr.startswith('y'):
		if tailstr == 'yd':
			return 'W'
		elif re.match(r'^(g?ogwydd|(t|d|th|nh)ramgwydd.*)$', partstr):
			return 'w'
		elif re.match(r'^(wy|gwy|frogwy|llugwy)$', partstr):
			return 'W'
		elif re.match(r'^(c|ch|g|ngh)$', leadstr) and re.match(r'^ymp.*$', tailstr):
			return 'w' # Cwympo: just a freaky exception
		elif re.match(r'^(c|ch|g|ngh)$', leadstr):
			return 'w' # Cwympo: just a freaky exception
		elif pre == 'g': # XXX and not soft mutated  OR  pre == '' or 'ng' and mutated:
			return 'W' # XXX fix with counterexamples below!
		elif pre == 'ch':
			return 'W'
		else:
			return 'W'
	elif is_simple_vowel_cluster(post):
		return 'w'
	elif is_simple_cons(post):
		return 'W'
	elif post == '$':
		return 'W'
	elif pre == '^' and post == 'i':
		return 'w'
	elif pre == 'y':
		if len(tailstr) >= 2 and tail[0] == 'i' and is_possible_vowel_cluster(tail[1]):
			return 'W'
		elif is_possible_vowel_cluster(post):
			return 'w'
		else:
			# XXX angen mwy o ymchwil: bywgraffiad, cywrain ... ?
			return 'W'
	elif (is_simple_vowel_cluster(pre) or pre == 'i') and (is_simple_vowel_cluster(post) or post == 'i'):
		return 'w'
	else:
		_logger.debug(f"get_type_of_w: This code should never be reached: {locals()}")


if __name__ == '__main__':
    badwords = set()
    while True:
        try:
            line = input()  # Python 3's input() returns str, which is Unicode
        except EOFError:
            break
        readings = []
        for word in re.findall(r'\w+', line):
            try:
                stressed_phones = get_stressed_phones(word)
            except (ValueError, TypeError):
                readings.append('??')
                badwords.add(word)
                traceback.print_exc(file=sys.stdout)
                continue
            readings.append(' '.join(stressed_phones))
        print(' | '.join(readings))

    if not badwords:
        sys.exit(0)
    print('Bad words: ' + ' '.join(badwords), file=sys.stderr)
    sys.exit(1)
