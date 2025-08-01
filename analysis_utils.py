import numpy as np

import editdistance as edd

import collections
from collections import defaultdict

def get_positional_letters(word):
    return [letter+str(i+1) for i,letter in enumerate(word)]

def get_non_positional_letters(letterlist):
    letter_index_dict = defaultdict(list)
    for letter in letterlist:
        letter_index_dict[letter[1:]] += [letter[0]]
    return letter_index_dict

def make_nonword(letter_index_dict,length,vocab,max_tries=1000,rng=None):
    '''tries to generate non-word of the specified length by sequentially choosing letters that are valid for each position'''
    rng = rng if not rng is None else np.random.default_rng()
    for i in range(max_tries):
        word = ""
        for j in range(length):
            word += rng.choice(np.array(letter_index_dict[str(j+1)]),size=1,replace=True)[0]
        if word not in vocab:
            return word
    raise Exception(f"attempted generating psuedoword {max_tries} times; all attempts yielded a word already in the vocab")


def make_nonword_list(letterlist,wordlist,n_pseudowords=None, rng=None):
    '''generates a list of nonwords with lengths drawn from the distribution of lengths in wordlist'''
    rng = rng if not rng is None else np.random.default_rng()
    if n_pseudowords is None:
        n_pseudowords=len(wordlist)
    letter_index_dict = get_non_positional_letters(letterlist)
    word_lengths, word_length_counts  = np.unique([len(w) for w in wordlist], return_counts=True)
    n_pseudowords=len(wordlist)
    psuedoword_lengths=[rng.choice(word_lengths, p=word_length_counts/sum(word_length_counts)) for _ in range(n_pseudowords)]
    psuedowords = [make_nonword(letter_index_dict,length,wordlist,rng=rng) for length in psuedoword_lengths] #maybe need to change so that psuedowords aren't repeated within this list
    return psuedowords






def integrate_interpolated_curve(x,y, axis=0): # perhaps I should make this more flexible so it can accept a single dt rather than an array of x values (in the case where we use equal time intervals)

    # print(x.shape)
    # print(y.shape)
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    bases = np.diff(x, axis=axis)#, prepend=x[0])
    assert np.all(bases>=0) # this function isn't really set up to handle x values that are not monotonically increasing in the 'x' array
    heights = np.diff(y, axis=axis)#, prepend=y[0])
    areas = (bases*heights/2) + (y[:-1]*bases)

    # prepend zeros because the area integrating from x=a to x=a is zero; this function essentially computes the approximate integral from x=x[0] to x=x[i] for i in range(len(x))
    return np.concatenate( [ np.zeros([size if ax_idx!=axis else 1 for ax_idx, size in enumerate(areas.shape)]),
                             np.cumsum(areas, axis=axis)
                            ], axis=axis 
                          )



def compute_best_nextbest_gaps(arr, axis=-1):
    sorted_activities = np.sort(arr, axis=axis)
    if axis<0:
        axis=len(arr.shape)+axis  # e.g. for 2D array, if axis=-1, 2+-1 = 1, if axis=-2, 2-2=0
    gap_slice = tuple(slice(None) if i!=axis else slice(-2,None) for i in range(len(arr.shape))) # on just the axis (node axis) take the best and next best activities which correspond to the -2 and -1 indices of the sorted activities array
    gaps = np.diff(sorted_activities[gap_slice],axis=axis) # gaps between the highest and second highest activities
    # print(gaps.shape)
    # print(sorted_activities[gap_slice].shape)
    # print(";;;;;;;;;;;;;;;;;;;;;;;;")
    return gaps

def compute_cumulative_gap_decision_indices(xHistory, times, stop_thresh=1, time_axis = 0, node_axis=-1, keepdims=True):


    gaps = compute_best_nextbest_gaps(xHistory, axis=node_axis) # gaps now has size 1 along the node_axis
    time_reshape = [1,]*len(gaps.shape)
    time_reshape[time_axis] = times.shape[0]

    cumulative_gaps = integrate_interpolated_curve(times.reshape(time_reshape), gaps, axis=time_axis)

    if isinstance(stop_thresh , type("")):
        if "pct" in stop_thresh:
            pct = float(stop_thresh.strip("pct"))
            # cumulative_gaps: (n_times, n_stimulus_words, 1)
            # max cumulative gap across time for each stimulus word
            stop_thresh = np.percentile(cumulative_gaps.max(axis=0) , pct)
            print(stop_thresh)
        elif "min" in stop_thresh:
            # print(cumulative_gaps.shape)
            stop_thresh = cumulative_gaps.max(axis=0).min()
    breaks_thresh = cumulative_gaps >= stop_thresh 

    decision_index_arr = np.where(breaks_thresh.any(axis=time_axis,keepdims=True), breaks_thresh.argmax(axis=time_axis, keepdims=True), -1) # decision_indices 
    if not keepdims:
        return decision_index_arr.squeeze(axis=(time_axis,node_axis))

    return decision_index_arr







def get_pairwise_edit_distance(*args):
    if len(args)==1:
        wordlists = [args[0],args[0]]
    elif len(args)==2:
        wordlists = [args[0], args[1]]
    else:
        raise Exception(f"'get_pairwise_edit_distance' function can only accept one or two wordlists, not {len(args)}")
    pairwise_edit_distance=np.zeros([len(wordlists[0]),len(wordlists[1])])
    for i in range(len(wordlists[0])):
        for j in range(len(wordlists[1])):
            pairwise_edit_distance[i,j]=edd.eval(wordlists[0][i],wordlists[1][j])
            # pairwise_edit_distance[j,i]=pairwise_edit_distance[i,j]
    return pairwise_edit_distance



