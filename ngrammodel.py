import string
import random
import time
import re 
import math
import numpy

def lin_reg(a,b):

    n = numpy.size(a)
  
    xbar = numpy.mean(a)
    ybar = numpy.mean(b)

    sumab = numpy.sum(b*a) - n*ybar*xbar
    sumaa = numpy.sum(a*a) - n*xbar*xbar

    c1 = sumab / sumaa
    c0 = ybar - c1*xbar
  
    return (c0, c1)

            #######################  GET SENTECNCES ####################
def get_sentences(a,b):
    sentences = []
    for i in range(a,b):
        with open('Book'+str(i)+'.txt', encoding="utf-8") as f:                 
            text = f.read()
            text = re.sub(r'[^\w\s\d\.\,]','',text)
            # text = re.sub('“','',text)
            text = re.sub(' +', ' ', text)
            text = re.sub(r'\n','',text)
            # text = re.sub('”','',text)
            text = text.replace('J . K . R O W L ! N G','JK Rowling')
            text = text.replace('J.K.','JK')
            text = text.replace('Mr.','Mr')
            text = text.replace('Mrs.','Mrs')
            text = text.lower()
            text = text.split('.')
            for sentence in text:
                # add back the fullstop
                if sentence.isspace() or sentence=='':
                    continue
                sentence += '.'
                sentences.append(sentence)
    return sentences

#tokenize function to delete punctuations 
# change the sentence into list of words
            #######################  TOKENIZE SENTECNCES ####################
def tokenize(sentence):
    # remove all punctuations
    for punct in string.punctuation:
        if punct == ',':
            sentence = sentence.replace(punct,' '+punct+' ')
        elif punct=='.':
            sentence = sentence.replace(punct,' '+punct+' ')
        else:
            sentence = sentence.replace(punct,'')
    t = sentence.split(' ')
    return t

# make Ngrams of a sentence 
def makeNgrams(n,sentences):
    l = []
    for sentence in sentences:
        
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        for i in range(n-1,len(tokenized_sentence)):
            prev_words = [tokenized_sentence[i-p-1] for p in range(n-2,-1,-1)]
            prev_words = '#'.join(prev_words)
            l.append((prev_words,tokenized_sentence[i]))

    return l


class Normal_Ngram:
    def __init__(self,n,sentences):
        self.n = n
        self.context = {}
        self.count_ngram = {}
        self.words_count = {}  #store count of each word in the text
        self.tot_words=0.0  
        self.tot_ngrams = 0.0
        self.update_context_count(sentences)
        self.update_words_count(sentences)
        self.freq_of_freq = {}
    
    def update_words_count(self,sentences):
        n = self.n
        for sentence in sentences:
            sep = tokenize(sentence)
            sep = (n-1)*['<s>']+sep
            for word in sep:
                if word in self.words_count:
                    self.words_count[word] += 1.0                           #update_words_count
                else:
                    self.words_count[word] = 1.0
                self.tot_words += 1.0
        return


    def update_context_count(self,sentences):
        n = self.n
        self.update_words_count(sentences)
        ngrams = makeNgrams(n,sentences)

        self.tot_ngrams  = len(ngrams)
                                                                                    #update_context_count
        for tup in ngrams:       
            prev,curr = tup

            if prev in self.context:
                self.context[prev].append(curr)                         
            else:
                self.context[prev] = [curr]

            if tup in self.count_ngram:
                self.count_ngram[tup] += 1.0
            else:
                self.count_ngram[tup] = 1.0
    

    def calc_prob_a1(self,context,target):
        # conditional probabibility of the target word given context

        n = self.n
        V = float(len(self.words_count))
        if n==1:
            try:
                prob = ((self.words_count[target]+1.0)/(self.tot_words+V))
                return prob
            except KeyError:
                prob = 1.0/V
                return prob
        count_context = 0.0
                                                                                        # calc_prob_a1
        try:
            count_context = float(len(self.context[context]))
        except KeyError:
            count_context = 0.0
        count_context_token = 0.0
        try: 
            count_context_token = self.count_ngram[(context,target)]
        except KeyError: 
            count_context_token = 0.0
        prob = (count_context_token+1.0)/(count_context+V)
        return prob
    
    def calc_prob(self,context,target):
        # conditional probabibility of the target word given context
        n = self.n
        if n==1:
            prob = (self.words_count[target]/self.tot_words)
            return prob
                                                                                        #calc_prob
        try:
            count_context = float(len(self.context[context]))
            count_context_token = self.count_ngram[(context,target)]
            prob = (count_context_token/count_context)
        except KeyError:
            prob = 0.0
        return prob

    def next_word_1(self):
        random.seed(time.time())
        r = random.random()
        p_w_p = {}
        for word in self.words_count:
            p_w_p[word] = self.calc_prob("",word)
        prob = 0.0                                                                  #next_Word_1
        for sorted_word in sorted(p_w_p):
            prob += p_w_p[sorted_word]
            if(prob>r):
                return sorted_word


    def next_word(self,context):
        # print(context)
        poss_words = []
        try:
            poss_words = self.context[context]
        except KeyError:
            print(context)


        
        pair_word_prob = {}
        for word in poss_words:
            pair_word_prob[word] = self.calc_prob(context,word)
                                                                                                # next_word
        random.seed(time.time())
        r = random.random()
        prob = 0.0
        
        # mn = 1
        for sorted_word in sorted(pair_word_prob):
            prob += pair_word_prob[sorted_word]
            if(prob>=r):
                # print(mn)
                return sorted_word
    
    def good_turing_ff(self):
        n = self.n
        for ct in self.count_ngram.values():
            if ct in self.freq_of_freq:
                self.freq_of_freq[ct]+=1
            else:
                self.freq_of_freq[ct]=1



                                    #CALCULATE GOOD-TURING PROBABILITY#

    def calc_prob_good_turing2(self,context,token):
        n = self.n
        fof = self.freq_of_freq
        lk = list(fof.keys())
        lv = list(fof.values())
        logc = numpy.log(lk)
        logNc = numpy.log(lv)
        w = lin_reg(logc,logNc)
        # w = [0.5, 0.5]

        a = w[0]
        b = w[1]
        # N = [0.0]*int(max(lk)+2)
        # for i in range (1,251):
        #     if i in fof:
        #         N[i] = fof[i]
        #     else:
        #         N[i] = math.exp(a+(b*math.log(i)))

        tct = self.tot_ngrams
        c = 0
        try:
            c = int(self.count_ngram[(context,token)])
        except KeyError:
            c = 0
        cstar = c
        Nc = 0.0
        Ncp1 = 0.0
        if c >=int(max(fof.keys())) :
            cstar = c
            Nc = fof[c]
        elif c==0:
            Nc = tct
            Ncp1 = fof[1]
            cstar = ((c+1)*Ncp1)/Nc
        else:
            if c in fof:
                Nc = fof[c]
            else:
                Nc = math.exp(a+(b*math.log(c)))
            if c+1 in fof:
                Ncp1 = fof[c+1]
            else:
                Ncp1 = math.exp(a+(b*math.log(c+1)))
            
            cstar = ((c+1)*Ncp1)/Nc
        
        # print(Nc,tct)
        logprob = math.log(cstar)-math.log(Nc)-math.log(tct)
        return logprob 


    def generate(self):
        n = self.n
        curr_context_v = (n-1)*['<s>']
                                                                                                # generate
        curr_context = '#'.join(curr_context_v)
        # print(curr_context,"id")
        ct = 0
        while True and ct<50:
            if n==1:
                s = self.next_word_1()
            else:
                s = self.next_word(curr_context)
            # print('s= ',s)
            if(s==','):
                print(s,end="")
            else:
                print(s,end=' ')
            if s=='.':
                break
            if n==1:         
                curr_context = ''
                ct+=1
                continue
            curr_context_v = curr_context.split('#')
            curr_context_v.append(s)
            curr_context_v.pop(0)
            curr_context = '#'.join(curr_context_v)
            ct+=1

        print()
        print()

    def test(self,mode):
        n = self.n
        
        test_sent = get_sentences(7,8)
        logsum = 0.0
        word_ct = 0.0
        it = 0
        for sentence in test_sent:
            tokenized_sentence = tokenize(sentence)
            tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
            word_ct += float(len(tokenized_sentence))
            it+=1
        
        ct =0
        for sentence in test_sent:
            # if(ct%1000==0):
            #     print(ct)
            
            tokenized_sentence = tokenize(sentence)
            tokenized_sentence = (n-1)*['<s>']+tokenized_sentence

            for i in range(n-1,len(tokenized_sentence)):
                prev_words = [tokenized_sentence[i-p-1] for p in range(n-2,-1,-1)]
                prev_words = '#'.join(prev_words)
                if mode == 'addone':
                    prob = self.calc_prob_a1(prev_words,tokenized_sentence[i])
                elif mode == 'goodturing':
                    prob = self.calc_prob_good_turing2(prev_words,tokenized_sentence[i])
                    logsum += prob
                    continue
                logsum += math.log(prob)
            
            ct+=1

        return math.exp((-1/word_ct) * logsum)    

def calc_stup_backoff(n,obj,context,token):
    mod = obj[n]
    if(n==1):
        return mod.calc_prob_a1(context,token)
    if mod.calc_prob(context,token)==0.0:
        v = context.split('#')
        v.pop(0)
        context = '#'.join(v)
        return ((0.4)* calc_stup_backoff(n-1,obj,context,token))
    else:
        return mod.calc_prob(context,token)

def test_backoff(n):
    test_sent = get_sentences(7,8)
    logsum = 0.0
    word_ct = 0.0
    it = 0
    for sentence in test_sent:
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        word_ct += float(len(tokenized_sentence))
        it+=1
    
    ct =0
    for sentence in test_sent:
        # if(ct%20==0):
        #     print(ct)
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        for i in range(n-1,len(tokenized_sentence)):
            prev_words = [tokenized_sentence[i-p-1] for p in range(n-2,-1,-1)]
            prev_words = '#'.join(prev_words)
            prob = calc_stup_backoff(n,obj,prev_words,tokenized_sentence[i])
            logsum += math.log(prob)
        
        ct+=1
    return math.exp((-1/word_ct) * logsum)    

def calc_prob_inter(n,obj,context,target):
    mod= obj[n]
    if(n==1):
        return mod.calc_prob_a1(context,target)

    prob = mod.calc_prob(context,target)
    v = context.split('#')
    v.pop(0)
    context = '#'.join(v)

    return (prob+ (n-1)*calc_prob_inter(n-1,obj,context,target))/n

def test_inter(n):
    test_sent = get_sentences(7,8)
    logsum = 0.0
    word_ct = 0.0
    it = 0
    for sentence in test_sent:
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        word_ct += float(len(tokenized_sentence))
        it+=1

    ct =0
    for sentence in test_sent:
        # if(ct%20==0):
        #     print(ct)
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        for i in range(n-1,len(tokenized_sentence)):
            prev_words = [tokenized_sentence[i-p-1] for p in range(n-2,-1,-1)]
            prev_words = '#'.join(prev_words)
            prob = calc_prob_inter(n,obj,prev_words,tokenized_sentence[i])
            logsum += math.log(prob)
        
        ct+=1
    return math.exp((-1/word_ct) * logsum)

dp = {}

Kneser_ney_dict = {}

def makeKneserNeydict(obj):
    for i in range(1,len(obj)):
        for tup in obj[i].count_ngram.keys():
            prev,curr = tup
            if (curr,i) in Kneser_ney_dict:
                Kneser_ney_dict[(curr,i)] += obj[i].count_ngram[tup]
            else:
                Kneser_ney_dict[(curr,i)] = 1.0


def calc_prob_kneser(n,obj,context, token,d):
    mod = obj[n]
    if (context,token) in dp:
        return dp[(context,token)]

    V = float(len(mod.words_count))
    lamb = 0
    if n == 1:
        prob_context_token = 0.0
        try: 
            prob_context_token = mod.words_count[token]/V
        except KeyError:
            prob_context_token = (20/V)*d

        return prob_context_token
    else:
        try:
            count_context = float(len(mod.context[context]))
            first_term = max(mod.count_ngram[(context,token)] - d, 0)/count_context
        except KeyError:
            first_term = 0.0
    try:
        ct = Kneser_ney_dict[(token,n)]
        count_context = float(len(mod.context[context]))
        lamb = (d*ct)/(count_context)
    except KeyError:
        return (20/V)*d
    if lamb==0:
        return (20/V)*d

    v = context.split('#')
    v.pop(0)
    context = '#'.join(v)
    sec_term = lamb*calc_prob_kneser(n-1,obj,context,token,d)
    dp[(context,token)] = first_term+sec_term
    return first_term + sec_term

def test_kneser(n):
    test_sent = get_sentences(7,8)
    makeKneserNeydict(obj)
    logsum = 0.0
    word_ct = 0.0
    it = 0
    for sentence in test_sent:
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        word_ct += float(len(tokenized_sentence))
        it+=1

    ct =0
    for sentence in test_sent:
        # if ct%20==0:
        #     print(ct)
        tokenized_sentence = tokenize(sentence)
        tokenized_sentence = (n-1)*['<s>']+tokenized_sentence
        for i in range(n-1,len(tokenized_sentence)):
            prev_words = [tokenized_sentence[i-p-1] for p in range(n-2,-1,-1)]
            prev_words = '#'.join(prev_words)
            prob = calc_prob_kneser(n,obj,prev_words,tokenized_sentence[i],0.6)
            logsum += math.log(prob)
        
        ct+=1
    return math.exp((-1/word_ct) * logsum)

all_sentences = get_sentences(1,8)
# print(makeNgrams(1,all_sentences))
# print(all_sentences)

obj = []

obj.append("00")
print('generating models ...')
for i in range(1,10):
    m = Normal_Ngram(i,all_sentences)
    print("generating sentence for: n = ",i, " :")
    m.generate()
    # obj.append(m)
    # m.good_turing_ff()
    # print(m.test('goodturing'))
    # print(m.freq_of_freq)
print('Model generated:-')

# print("Smoothing method == Stupid_backoff:")
# for i in range(1,8):
#     print("Perplexity for n = ",i)
    # obj[i].good_turing_ff()
    # print("Smoothing method == Addone/Laplace: ")
    # print(obj[i].test('addone'))
    # print(obj[i].test('goodturing'))
    # print(test_inter(i))
    # print(test_backoff(i))
    # print(test_kneser(i))


