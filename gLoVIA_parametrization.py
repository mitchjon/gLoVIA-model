import numpy as np
# from scipy import integrate




class gLoVIA_Parameters:
    def __init__(self, wordlist, node_sort = 'alphabetical', rng=None):
        self.wordlist = wordlist
        self.letterlist = []
        self.nodelist = None
        self.index_matrix = None
        self.fn = lambda n: np.log(n)/np.log(10) # used to scale parameter values with the log of the network size

        #----------------attributes assigned for later convenience---------------
        self.n_words = None
        self.n_letters = None
        self.n_nodes = None
        self.pattern_matrix = None
        self.rate_matrix = None

        #----------------------Parameters of the Model-------------------------
        self.growth_rates = None
        self.interaction_matrix = None




        #------------------end of attribute initialization----------------------
        #the following code assigns variables to some of the attributes previously initialized to None

        #---------create letterlist and index_matrix simultaniously -----------
        #index_matrix has columns coresponding to words in the same index of wordlist
        #each of these columns holds a boolian index array for the letterlist,
        #giving the index of the letters in the word corresponding to this column
        max_node_num = len(''.join(wordlist)) + len(wordlist)
        index_matrix = np.zeros( (max_node_num,len(wordlist)) , dtype = 'bool')
        for i, word in enumerate(wordlist):
            for j,letter in enumerate(word):
                positional_letter = letter + str(j+1)
                if positional_letter in self.letterlist:
                    index_matrix[self.letterlist.index(positional_letter),i] = True
                else:
                    index_matrix[len(self.letterlist), i ] = True
                    self.letterlist.append(positional_letter)

        #---------------sort word/letter lists if specified-------------------

        #if we wish to sort the wordlists/letterlists in any way, the following
        #code generate the indices to sort by, and then apply that sort to both
        #the columns/rows of the index_matrix, and to the associated word/letter lists.
        if node_sort == 'alphabetical':
            word_sort_indices = np.argsort(wordlist)
            letter_sort_indices = np.argsort(self.letterlist)
        elif node_sort == 'permutation':
            if rng is None:
                rng = np.random.default_rng()
            word_sort_indices = rng.choice(range(len(self.wordlist)), size=len(self.wordlist), replace=False)
            letter_sort_indices = rng.choice(range(len(self.letterlist)), size=len(self.letterlist), replace=False)
        elif node_sort == None:
            word_sort_indices = list(range(len(self.wordlist)))
            letter_sort_indices = list(range(len(self.letterlist)))
        else:
            raise Exception('node sort method misspelled or incorrectly passed into __init__ for Model')

        self.wordlist = [self.wordlist[index] for index in word_sort_indices]
        self.letterlist = [self.letterlist[index] for index in letter_sort_indices]
        #sort index_matrix by rows, then by columns
        self.index_matrix = index_matrix[letter_sort_indices,:][:,word_sort_indices]


        #--------------assigning attributes for convenience--------------
        self.nodelist = self.letterlist + self.wordlist
        self.n_words = len(self.wordlist)
        self.n_letters = len(self.letterlist)
        self.n_nodes = len(self.nodelist)
        #---------------end of __init__------------------------------


    def set_pattern_generator(self, pattern_on=None, pattern_off=None):

        def create_pattern_matrix():
            #apped the identity matrix for the w-w interaction
            P = np.append(self.index_matrix,np.identity(self.n_words), axis = 0)
            #where P is 1, pattern_matrix = on; where P = 0, pattern_matrix = off
            P = P*(pattern_on-pattern_off) + P*pattern_off
            self.pattern_matrix = P
            return P

        self.create_pattern_matrix = create_pattern_matrix



    def set_r(self, lgrowth, wgrowth):
        l1 = len(self.letterlist)
        l2 = len(self.wordlist)
        self.r = np.array(l1*[lgrowth] + l2*[wgrowth]) 

        return self.r


    def set_M_method1(self):
        P = self.create_pattern_matrix()
        M = -np.einsum('n,m->nm',  self.r, np.linalg.pinv(P).sum(axis=0))

        self.interaction_matrix = M
        self.M=M

        return M




    def set_M_method2(self, ldecay=-1,wdecay=-1,ltl=-1,wtw=-1):

        n = len(self.nodelist)

        M = np.zeros((n, n))

        fnv=self.fn(n)
        ldecay_adj = ldecay*fnv
        wdecay_adj = wdecay*fnv
        ltl_adj = ltl*fnv
        wtw_adj = wtw*fnv

        # latteral inhibition (ltl) and self-inhibition (ldecay) in letter layer
        A = np.zeros(( len(self.letterlist) , len(self.letterlist) ))
        for i, a in enumerate(self.letterlist):
            for j, b in enumerate(self.letterlist):
                if i == j:
                    A[i,j] = ldecay_adj
                else:
                    if a[1:] == b[1:]:
                        A[i,j] = ltl_adj


        #Build C (letter to word) and B (word to letter)
        P = self.create_pattern_matrix()
        lMat = P[:len(self.letterlist),:]
        wMat = P[-len(self.wordlist):,:]
        C = wMat@np.linalg.pinv(lMat)   #llxlw
        B = C.T                         #lwxll

        # latteral inhibition (wtw) and self-inhibition (wdecay) in word layer
        D = np.zeros((len(self.wordlist), len(self.wordlist)))
        for i, a in enumerate(self.wordlist):
            for j, b in enumerate(self.wordlist):
                if i != j:
                    D[i,j] = wtw_adj
                else:
                    D[i,j] = wdecay_adj
        M[:len(self.letterlist), :len(self.letterlist)] = A
        M[:len(self.letterlist), -len(self.wordlist):] = B
        M[-len(self.wordlist):, :len(self.letterlist)] = C
        M[-len(self.wordlist):, -len(self.wordlist):] = D
        self.interaction_matrix = M
        self.M=M

        return M



    def set_M_method3(self,eps=0.01,eps2=0.1):
        import matplotlib.pyplot as plt
        M = np.zeros((self.n_nodes, self.n_nodes))

        P = self.create_pattern_matrix()
        lMat = P[:self.n_letters,:]
        wMat = P[-self.n_words:,:]



        # build A: letter to letter
        A = -eps2*np.ones((len(self.letterlist),len(self.letterlist)))
        #A s.t. it has lMat for eigenvectors, and levMat are the eigenvalues
        levMat = 1+np.random.randn(len(self.wordlist))*eps
        levMat = np.diag(levMat)
        # MV = VD ==> M = VDV*, so compute VDV*
        A += lMat @ levMat @ np.linalg.pinv(lMat)


        # build B: word to letter, and C: letter to word
        C = wMat@np.linalg.pinv(lMat)   #llxlw
        B = C.T                         #lwxll

        # build D: word to word
        D = -eps2*np.ones(( len(self.wordlist) , len(self.wordlist) ))
        #D s.t. it has wMat for eigenvectors, and wevMat are the eigenvalues
        wevMat = 1+np.random.randn(len(self.wordlist))*eps
        wevMat = np.diag(wevMat)
        # MV = VD ==> M = VDV*, so compute VDV*
        D += wMat @ wevMat @ np.linalg.pinv(wMat)

        M[:len(self.letterlist), :len(self.letterlist)] = A
        M[:len(self.letterlist), -len(self.wordlist):] = B
        M[-len(self.wordlist):, :len(self.letterlist)] = C
        M[-len(self.wordlist):, -len(self.wordlist):] = D
        self.interaction_matrix = M
        self.M=M
        return M



    def set_M_method4(self, ldecay = -1,wdecay = -1, ltl = -0.9,wtw = -0.5, ltw = 0.2, wtl = 0.2):

        n = len(self.nodelist)
        M = np.zeros((n, n))

        fnv=self.fn(n)

        # adjusted parameter values for network size 'n'
        ldecay_adj = ldecay*fnv
        wdecay_adj = wdecay*fnv
        ltl_adj = ltl*fnv
        wtw_adj = wtw*fnv
        ltw_adj = ltw*fnv
        wtl_adj = wtl*fnv



        # latteral inhibition (ltl) and self-inhibition (ldecay) in letter layer
        A = np.zeros(( len(self.letterlist) , len(self.letterlist) ))
        for i, a in enumerate(self.letterlist):
            for j, b in enumerate(self.letterlist):
                if i == j:
                    A[i,j] = ldecay_adj
                else:
                    if a[1:] == b[1:]:
                        A[i,j] = ltl_adj


        # between-layer excitation: word to letter (wtl) and letter to word (ltw)
        B = np.zeros( (len(self.letterlist),len(self.wordlist) ))
        C = np.zeros( (len(self.letterlist),len(self.wordlist) )).T
        for i, a in enumerate(self.letterlist):
            for j, b in enumerate(self.wordlist):
                word_letters = [l+str(li+1) for li,l in enumerate(b)]
                if a in word_letters:
                    B[i,j] = wtl_adj
                    C[j,i] = ltw_adj



        # latteral inhibition (wtw) and self-inhibition (wdecay) in word layer
        D = np.zeros((len(self.wordlist), len(self.wordlist)))
        for i, a in enumerate(self.wordlist):
            for j, b in enumerate(self.wordlist):
                if i != j:
                    D[i,j] = wtw_adj
                else:
                    D[i,j] = wdecay_adj
        M[:len(self.letterlist), :len(self.letterlist)] = A
        M[:len(self.letterlist), -len(self.wordlist):] = B
        M[-len(self.wordlist):, :len(self.letterlist)] = C
        M[-len(self.wordlist):, -len(self.wordlist):] = D
        self.interaction_matrix = M
        self.M=M
        return M






    def set_xZero_generator(self, xZero_on=1, xZero_off=1e-4):

        def xZero_generator(input_word):
            xZero = np.ones(( len(self.nodelist) , ))*xZero_off
            positional_letters = [l+str(i+1) for i,l in enumerate(input_word)]
            positional_letter_indices = [self.letterlist.index(l) for l in positional_letters]
            xZero[positional_letter_indices] += (xZero_on-xZero_off)
            return xZero

        self.xZero_generator=xZero_generator
        return xZero_generator


    # def odeint(self, dxdt, xZero, times):
    #     return integrate.odeint(dxdt, xZero, times)

    def euler_integrate(self, dxdt, xZero, times):
        xHistory = np.zeros(times.shape + xZero.shape)
        # dxdthistory = np.zeros(times.shape + xZero.shape)
        xHistory[0] = xZero
        for i,dt in enumerate(np.diff(times)):
            # dxdthistory[i+1] = dxdt(xHistory[i],times[i]+dt)*dt
            xHistory[i+1] = xHistory[i] + dxdt(xHistory[i],times[i]+dt)*dt
        return xHistory

    def batch_integrate(self,times,input_words):

        r=self.r
        M = self.interaction_matrix
        batch_size = len(input_words)

        def dudt(u,t_i):
            rest = 0.0

            # the input should originally have shape (batch_size, n_nodes), but must be flattened before integrated
            u_orig = u.reshape(batch_size, -1)
            r_cast = r[None,:]
            dudt_unflat = r_cast + np.exp(u_orig)@(M.T) - rest*r_cast*np.exp(-u_orig)#(batch_size,n_nodes)@(n_nodes,n_nodes) -> (batch_size,n_nodes)
            return dudt_unflat.reshape(-1)

        xZero = np.vstack([self.xZero_generator(input_word) for input_word in input_words]).reshape(-1)
        uZero = np.log(xZero)
        uHistory = self.euler_integrate( dudt , uZero , times )
        uHistory = uHistory.reshape(uHistory.shape[0],batch_size,-1).transpose(0,2,1)
        xHistory = np.exp(uHistory)
        return xHistory


    def batch_integrate_IA(self,times,input_words):

        r=self.r
        M = self.interaction_matrix
        batch_size = len(input_words)

        def Relu(x):
            return np.where(x>=0, x, 0)

        def dxdt(x,t_i):

            minval = 0
            maxval = 1
            rest = 0

            x[x<minval] = minval
            x[x>maxval] = maxval
            xclamped = x
            # the input should originally have shape (batch_size, n_nodes), but must be flattened before integrated
            x_orig = xclamped.reshape(batch_size, -1)
            r_cast = -r[None,:]


            # maxval = 2*x_orig

            #(batch_size,n_nodes)@(n_nodes,n_nodes) -> (batch_size,n_nodes)
            dxdt_unflat =   (maxval - x_orig) * Relu(Relu(x_orig)@(M.T)) \
                          + (minval - x_orig) * Relu( - Relu(x_orig)@(M.T) ) \
                          - x_orig*r_cast + rest*r_cast
            return dxdt_unflat.reshape(-1)

        xZero = np.vstack([self.xZero_generator(input_word) for input_word in input_words]).reshape(-1)
        xHistory = self.euler_integrate( dxdt , xZero , times )
        xHistory = xHistory.reshape(xHistory.shape[0],batch_size,-1).transpose(0,2,1)

        return xHistory


