import numpy as np
import gLV_IA as glv
import os
import pickle
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt





'''kbutil------------------------------------------------------------------------------------------'''

import pylab
from numpy import ceil,log2,histogram,abs,linspace,zeros,inf,log,vstack
from numpy import min as npmin
from numpy import max as mpmax
from numpy.random import randn
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
from matplotlib import colors,cm

import seaborn as sns

#_colors = ('k','r','orange','gold','g','b','purple','magenta',
#           'firebrick','coral','limegreen','dodgerblue','indigo','orchid',
#           'tomato','darkorange','greenyellow','darkgreen','yellow','deepskyblue','indigo','deeppink')
#_colors = ('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
#            '#f032e6', '#bcf60c','#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
#            '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000')
_colors = ('#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
            '#f032e6', '#bcf60c','#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
            '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#ffe119')
_symbols = ('o','s','^','<','>','x','D','h','p')
_lines = ('-','--','-.',':')

def color_wheel(colors=_colors,symbols=_symbols,lines=_lines):
    """
    Returns a generator that cycles through a selection of colors,symbols,
    lines styles for matplotlib.plot.  Thanks to Ryan Gutenkunst for this
    idiom.
    """
    if not colors:
        colors = ('',)
    if not symbols:
        symbols = ('',)
    if not lines:
        lines = ('',)

    while 1:
        for l in lines:
            for s in symbols:
                for c in colors:
                    yield (c,s,l)



def pylab_pretty_plot(lines=2,width=3,size=4,labelsize=16,markersize=10,fontsize=20,lfontsize=16,lframeon=False,usetex=True):
    """
    Changes pylab plot defaults to get nicer plots - frame size, marker size, etc.

    Parameters:
    ------------
    lines      : linewidth
    width      : width of framelines and tickmarks
    size       : tick mark length
    labelsize  : font size of ticklabels
    markersize : size of plotting markers
    fontsize   : size of font for axes labels
    lfontsize  : legend fontsize
    usetex     : use latex for labels/text?

    """
    pylab.rc("lines",linewidth=lines)
    pylab.rc("lines",markeredgewidth=size/3)
    pylab.rc("lines",markersize=markersize)
    pylab.rc("ytick",labelsize=labelsize)
    pylab.rc("ytick.major",pad=size)
    pylab.rc("ytick.minor",pad=size)
    pylab.rc("ytick.major",size=size*1.8)
    pylab.rc("ytick.minor",size=size)
    pylab.rc("xtick",labelsize=labelsize)
    pylab.rc("xtick.major",pad=size)
    pylab.rc("xtick.minor",pad=size)
    pylab.rc("xtick.major",size=size*1.8)
    pylab.rc("xtick.minor",size=size)
    pylab.rc("axes",linewidth=width)
    # pylab.rc("axes", labelpad=50)
    #pylab.rc("text",usetex=usetex)
    pylab.rc("font",size=fontsize)
    pylab.rc("legend",fontsize=lfontsize)
    pylab.rc("legend",frameon=lframeon)
    # plt.tight_layout()
'''--------------------------------------------------------------------------------------------------------'''






vocab = ['a', 'able', 'about', 'above', 'across', 'act', 'actor', 'active', 'activity', 'add', 'afraid', 'after', 'again', 'age', 'ago', 'agree', 'air', 'all', 'alone', 'along', 'already', 'always', 'am', 'amount', 'an', 'and', 'angry', 'another', 'answer', 'any', 'anyone', 'anything', 'anytime', 'appear', 'apple', 'are', 'area', 'arm', 'army', 'around', 'arrive', 'art', 'as', 'ask', 'at', 'attack', 'aunt', 'autumn', 'away', 'baby', 'base', 'back', 'bad', 'bag', 'ball', 'bank', 'basket', 'bath', 'be', 'bean', 'bear', 'beautiful', 'beer', 'bed', 'bedroom', 'behave', 'before', 'begin', 'behind', 'bell', 'below', 'besides', 'best', 'better', 'between', 'big', 'bird', 'birth', 'birthday', 'bit', 'bite', 'black', 'bleed', 'block', 'blood', 'blow', 'blue', 'board', 'boat', 'body', 'boil', 'bone', 'book', 'border', 'born', 'borrow', 'both', 'bottle', 'bottom', 'bowl', 'box', 'boy', 'branch', 'brave', 'bread', 'break', 'breakfast', 'breathe', 'bridge', 'bright', 'bring', 'brother', 'brown', 'brush', 'build', 'burn', 'business', 'bus', 'busy', 'but', 'buy', 'by', 'bundle', 'cake', 'call', 'can', 'candle', 'cap', 'car', 'card', 'care', 'careful', 'careless', 'carry', 'case', 'cat', 'catch', 'central', 'century', 'certain', 'chair', 'chance', 'change', 'chase', 'cheap', 'cheese', 'chicken', 'child', 'children', 'chocolate', 'choice', 'choose', 'circle', 'city', 'class', 'clever', 'clean', 'clear', 'climb', 'clock', 'cloth', 'clothes', 'cloud', 'cloudy', 'close', 'coffee', 'coat', 'coin', 'cold', 'collect', 'colour', 'comb', 'come' , 'comfortable', 'common', 'compare', 'complete', 'computer', 'condition', 'continue', 'control', 'cook', 'cool', 'copper', 'corn', 'corner', 'correct', 'cost', 'contain', 'count', 'country', 'course', 'cover', 'crash', 'cross', 'cry', 'cup', 'cupboard', 'cut', 'dance', 'danger', 'dangerous', 'dark', 'daughter', 'day', 'dead', 'decide', 'decrease', 'deep', 'deer', 'depend', 'desk', 'destroy', 'develop', 'die', 'different', 'difficult', 'dinner', 'direction', 'dirty', 'discover', 'dish', 'do', 'dog', 'door', 'double', 'down', 'draw', 'dream', 'dress', 'drink', 'drive', 'drop', 'dry', 'duck', 'dust', 'duty', 'destroy', 'dedicated', 'each', 'ear', 'early', 'earn', 'earth', 'east', 'easy', 'eat', 'education', 'effect', 'egg', 'eight', 'either', 'electric', 'elephant', 'else', 'empty', 'end', 'enemy', 'enjoy', 'enough', 'enter', 'equal', 'entrance', 'escape', 'even', 'evening', 'event', 'ever', 'every', 'everyone', 'exact', 'everybody', 'examination', 'example', 'except', 'excited', 'exercise', 'expect', 'expensive', 'explain', 'extremely', 'eye', 'face', 'fact', 'fail', 'fall', 'false', 'family', 'famous', 'far', 'farm', 'father', 'fast', 'fat', 'fault', 'fear', 'feed', 'feel', 'female', 'fever', 'few', 'fight', 'fill', 'film', 'find', 'fine', 'finger', 'finish', 'fire', 'first', 'fit', 'five', 'fix', 'flag', 'flat', 'float', 'floor', 'flour', 'flower', 'fly', 'fold', 'food', 'fool', 'foot', 'football', 'for', 'force', 'foreign', 'forest', 'forget', 'forgive', 'fork', 'form', 'fox', 'four', 'free', 'freedom', 'freeze', 'fresh', 'friend', 'friendly', 'from', 'front', 'fruit', 'full', 'fun', 'funny', 'furniture', 'further', 'future', 'game', 'garden', 'gate', 'general', 'gentleman', 'get', 'gift', 'give', 'glad', 'glass', 'go', 'goat', 'god', 'gold', 'good', 'goodbye', 'grandfather', 'grandmother', 'grass', 'grave', 'great', 'green', 'grey', 'ground', 'group', 'grow', 'gun', 'hair', 'half', 'hall', 'hammer', 'hand', 'happen', 'happy', 'hard', 'hat', 'hate', 'have', 'he', 'head', 'healthy', 'hear', 'heavy', 'hello', 'help', 'heart', 'heaven', 'height', 'hen', 'her', 'here', 'hers', 'hide', 'high', 'hill', 'him', 'his', 'hit', 'hobby', 'hold', 'hole', 'holiday', 'home', 'hope', 'horse', 'hospital', 'hot', 'hotel', 'house', 'how', 'hundred', 'hungry', 'hour', 'hurry', 'husband', 'hurt', 'I', 'ice', 'idea', 'if', 'important', 'in', 'increase', 'inside', 'into', 'introduce', 'invent', 'iron', 'invite', 'is', 'island', 'it', 'its', 'job', 'join', 'juice', 'jump', 'just', 'keep', 'key', 'kid', 'kill', 'kind', 'king', 'kitchen', 'knee', 'knife', 'knock', 'know', 'ladder', 'lady', 'lamp', 'land', 'large', 'last', 'late', 'lately', 'laugh', 'lazy', 'lead', 'leaf', 'learn', 'leave', 'leg', 'left', 'lend', 'length', 'less', 'lesson', 'let', 'letter', 'library', 'lie', 'life', 'light', 'like', 'lion', 'lip', 'list', 'listen', 'little', 'live', 'lock', 'lonely', 'long', 'look', 'lose', 'lot', 'love', 'low', 'lower', 'luck', 'machine', 'main', 'make', 'male', 'man', 'many', 'map', 'mark', 'market', 'marry', 'matter', 'may', 'me', 'meal', 'mean', 'measure', 'meat', 'medicine', 'meet', 'member', 'mention', 'method', 'middle', 'milk', 'mill', 'million', 'mind', 'mine', 'minute', 'miss', 'mistake', 'mix', 'model', 'modern', 'moment', 'money', 'monkey', 'month', 'moon', 'more', 'morning', 'most', 'mother', 'mountain', 'mouse', 'mouth', 'move', 'much', 'music', 'must', 'my', 'name', 'narrow', 'nation', 'nature', 'near', 'nearly', 'neck', 'need', 'needle', 'neighbour', 'neither', 'net', 'never', 'new', 'news', 'newspaper', 'next', 'nice', 'night', 'nine', 'no', 'noble', 'noise', 'none', 'nor', 'north', 'nose', 'not', 'nothing', 'notice', 'now', 'number', 'obey', 'object', 'ocean', 'of', 'off', 'offer', 'office', 'often', 'oil', 'old', 'on', 'one', 'only', 'open', 'opposite', 'or', 'orange', 'order', 'other', 'our', 'out', 'outside', 'over', 'own', 'page', 'pain', 'paint', 'pair', 'pan', 'paper', 'parent', 'park', 'part', 'partner', 'party', 'pass', 'past', 'path', 'pay', 'peace', 'pen', 'pencil', 'people', 'pepper', 'per', 'perfect', 'period', 'person', 'petrol', 'photograph', 'piano', 'pick', 'picture', 'piece', 'pig', 'pill', 'pin', 'pink', 'place', 'plane', 'plant', 'plastic', 'plate', 'play', 'please', 'pleased', 'plenty', 'pocket', 'point', 'poison', 'police', 'polite', 'pool', 'poor', 'popular', 'position', 'possible', 'potato', 'pour', 'power', 'present', 'press', 'pretty', 'prevent', 'price', 'prince', 'prison', 'private', 'prize', 'probably', 'problem', 'produce', 'promise', 'proper', 'protect', 'provide', 'public', 'pull', 'punish', 'pupil', 'push', 'put', 'queen', 'question', 'quick', 'quiet', 'quite', 'radio', 'rain', 'rainy', 'raise', 'reach', 'read', 'ready', 'real', 'really', 'receive', 'record', 'red', 'remember', 'remind', 'remove', 'rent', 'repair', 'repeat', 'reply', 'report', 'rest', 'restaurant', 'result', 'return', 'rice', 'rich', 'ride', 'right', 'ring', 'rise', 'road', 'rob', 'rock', 'room', 'round', 'rubber', 'rude', 'rule', 'ruler', 'run', 'rush', 'sad', 'safe', 'sail', 'salt', 'same', 'sand', 'save', 'say', 'school', 'science', 'scissors', 'search', 'seat', 'second', 'see', 'seem', 'sell', 'send', 'sentence', 'serve', 'seven', 'several', 'sex', 'shade', 'shadow', 'shake', 'shape', 'share', 'sharp', 'she', 'sheep', 'sheet', 'shelf', 'shine', 'ship', 'shirt', 'shoe', 'shoot', 'shop', 'short', 'should', 'shoulder', 'shout', 'show', 'sick', 'side', 'signal', 'silence', 'silly', 'silver', 'similar', 'simple', 'single', 'since', 'sing', 'sink', 'sister', 'sit', 'six', 'size', 'skill', 'skin', 'skirt', 'sky', 'sleep', 'slip', 'slow', 'small', 'smell', 'smile', 'smoke', 'snow', 'so', 'soap', 'sock', 'soft', 'some', 'someone', 'something', 'sometimes', 'son', 'soon', 'sorry', 'sound', 'soup', 'south', 'space', 'speak', 'special', 'speed', 'spell', 'spend', 'spoon', 'sport', 'spread', 'spring', 'square', 'stamp', 'stand', 'star', 'start', 'station', 'stay', 'steal', 'steam', 'step', 'still', 'stomach', 'stone', 'stop', 'store', 'storm', 'story', 'strange', 'street', 'strong', 'structure', 'student', 'study', 'stupid', 'subject', 'substance', 'successful', 'such', 'sudden', 'sugar', 'suitable', 'summer', 'sun', 'sunny', 'support', 'sure', 'surprise', 'sweet', 'swim', 'sword', 'table', 'take', 'talk', 'tall', 'taste', 'taxi', 'tea', 'teach', 'team', 'tear', 'telephone', 'television', 'tell', 'ten', 'tennis', 'terrible', 'test', 'than', 'that', 'the', 'their', 'theirs', 'then', 'there', 'therefore', 'these', 'thick', 'thin', 'thing', 'think', 'third', 'this', 'those', 'though', 'threat', 'three', 'tidy', 'tie', 'title', 'to', 'today', 'toe', 'together', 'tomorrow', 'tonight', 'too', 'tool', 'tooth', 'top', 'total', 'touch', 'town', 'train', 'tram', 'travel', 'tree', 'trouble', 'true', 'trust', 'twice', 'try', 'turn', 'type', 'uncle', 'under', 'understand', 'unit', 'until', 'up', 'use', 'useful', 'usual', 'usually', 'vegetable', 'very', 'village', 'voice', 'visit', 'value', 'vacuum', 'vampire', 'verb', 'validation', 'wait', 'wake', 'walk', 'want', 'warm', 'wash', 'waste', 'watch', 'water', 'way', 'we', 'weak', 'wear', 'weather', 'wedding', 'week', 'weight', 'welcome', 'well', 'west', 'wet', 'what', 'wheel', 'when', 'where', 'which', 'while', 'white', 'who', 'why', 'wide', 'wife', 'wild', 'will', 'win', 'wind', 'window', 'wine', 'winter', 'wire', 'wise', 'wish', 'with', 'without', 'woman', 'wonder', 'word', 'work', 'world', 'worry', 'worst', 'write', 'wrong', 'year', 'yellow', 'yes', 'yesterday', 'yet', 'you', 'young', 'your', 'yours', 'zero', 'zoo', 'zoom']
vocab=list(set(vocab))





analysis = glv.plotting.general_analysis_06_18_2025

def full_output_for_params_old(vocab,param_dict,t, pct=1, rng=None, filter_kwargs={"high_activity_factor":100000}): #high_activity_factor for filtering out trails where activities are unstable or get really large
    if rng is None:
        rng = np.random.default_rng()
    # rng_voc = np.random.default_rng()
    voc_size = param_dict["vocab_size"]
    vocab = rng.choice(vocab,size=voc_size,replace=False)

    model_params = glv.model_parameters.Parameters(vocab)
    sims_dir_name = "method_param_sims10000"


    model_params.set_pattern_generator(**param_dict["pattern_args"])
    if param_dict["r_args"]["wgrowth"] is None: #forcing wgrowth to be the same as lgrowth for None
        param_dict["r_args"]["wgrowth"] = param_dict["r_args"]["lgrowth"]
    r = model_params.set_r(**param_dict["r_args"])
    # print(param_dict)
    try:
        M_method = param_dict["M_args"].pop("method")
    except:
        print(param_dict["M_args"])
        raise

    if M_method =="m1":
        M = model_params.set_M_method1()
    elif M_method =="m2":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        M = model_params.set_M_method2( **param_dict["M_args"] )

    elif M_method =="m3":
        M = model_params.set_M_method3( **param_dict["M_args"] )
    elif M_method =="m4":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        if "ltw" not in param_dict["M_args"].keys():
            param_dict["M_args"]["ltw"] = param_dict["M_args"]["wtl"]

        M = model_params.set_M_method4( **param_dict["M_args"] )
    elif M_method =="m5":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]

        M = model_params.set_M_method5( **param_dict["M_args"] )
    else:
        # mmethod=param_dict["M_args"]["method"]
        raise Exception(f"parameters provided for 'set_M' should be chosen from ['m1', 'm2' ,'m3', 'm4'] (not '{M_method}')")


    # plt.imshow(M)
    # plt.show()
    model_params.set_xZero_generator(**param_dict["xZero_args"])
    if param_dict["integrator_args"]["forcing_type"] != "n":
        model_params.set_forcing_generator(**param_dict["forcing_args"])
    integrator = model_params.set_integrator(t,forcing_type=param_dict["integrator_args"]["forcing_type"])






    model_size=len(model_params.nodelist)
    # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # accumulated_thresh=0.0001/np.log(len(model_params.nodelist))#/np.log(80)

    vocab = model_params.wordlist


    xHistories, filter_mask = analysis.get_xHistories(model_params, t, vocab, filter_condition=analysis.filter_high_activations_by_median, filter_kwargs=filter_kwargs)
    # xHistories = xHistories[:,:,filter_mask]
    decisions, decision_indices, correct_decisions,decision_times = analysis.get_accuracy_flexible(model_params, t, vocab, pct=99, xHistories=xHistories, compute_xHistories=False)

    decision_indices = analysis.compute_cumulative_gap_decision_indices(xHistories, t, stop_thresh=.05, time_axis = 0, node_axis=-1, keepdims=False)







    edit_distance_activities = analysis.compute_lexical_competition_data(model_params, vocab, xHistories.transpose(0,2,1)[:,:,-len(model_params.wordlist):], edit_dists = 'all')

    # decision_indices = decision_indices[decision_indices!=-1] # (n_stim_words,) -> (n_decisions,)
    # print(decision_indices)
    # print([d%1 == 0 for d in decision_indices])
    # decision_indices 



    # filtered_ed_dist_act = {dist: {stim_word:activities for stim_word,activities in edit_distance_activities[dist].items() if vocab.index(stim_word) in decision_indices} for dist in edit_distance_activities.keys()}
    # print({dist: {stim_word:activities.shape for stim_word,activities in edit_distance_activities[dist].items() if decision_indices[vocab.index(stim_word)] != -1} for dist in edit_distance_activities.keys()})
    # print({dist: {stim_word:decision_indices[vocab.index(stim_word)] for stim_word,activities in edit_distance_activities[dist].items() if decision_indices[vocab.index(stim_word)] != -1} for dist in edit_distance_activities.keys()})


    filtered_ed_dist_act = {dist: {stim_word:activities.mean(axis=1,keepdims=True) for stim_word,activities in edit_distance_activities[dist].items() if decision_indices[vocab.index(stim_word)] != -1 and correct_decisions[vocab.index(stim_word)] and activities.shape[1]>0} for dist in edit_distance_activities.keys()}
    edit_distance_activities = filtered_ed_dist_act
    if any([len(list(edit_distance_activities[k].values()))==0 for k in edit_distance_activities.keys()]):
        return -1, -1,-1,-1,int(np.sum(filter_mask))
        raise Exception("--------------kkdsdf")
    edit_dist0_arr = np.hstack( list(edit_distance_activities[0].values()) ) #[decision_indices,:]
    edit_dist1_arr = np.hstack( list(edit_distance_activities[1].values()) ) #[decision_indices,:]
    edit_dist2_arr = np.hstack( list(edit_distance_activities[2].values()) ) #[decision_indices,:] 
    edit_dist_unrel_arr = np.hstack([ed_dist_act for d in edit_distance_activities.keys()
                                    for ed_dist_act in edit_distance_activities[d].values() if not d in [0,1,2] ]) # [decision_indices,:] #making a flat list for all 




    # print("Edit distance order is correct:  ----------------------")
    # print(edit_dist0_arr.mean()>edit_dist1_arr.mean() and edit_dist1_arr.mean()>edit_dist2_arr.mean() and edit_dist2_arr.mean()>edit_dist_unrel_arr.mean())

    filtered_decision_times = {dist: {stim_word:t[decision_indices[vocab.index(stim_word)]] for stim_word,activities in edit_distance_activities[dist].items() if decision_indices[vocab.index(stim_word)] != -1 and activities.shape[1]>0} for dist in edit_distance_activities.keys()}
    mdt = np.mean([vi for vo in filtered_decision_times.values() for vi in vo.values()])
    mdti = np.argmin(np.abs( t - mdt )) # the time in t that is closest to the mean decision time should be the smallest element of abs(t-mean(decision_times) )

    # print(edit_dist0_arr[mdti].mean()>edit_dist1_arr[mdti].mean() and edit_dist1_arr[mdti].mean()>edit_dist2_arr[mdti].mean() and edit_dist2_arr[mdti].mean()>edit_dist_unrel_arr[mdti].mean())
    correct_edit_dist = edit_dist0_arr[mdti].mean()>edit_dist1_arr[mdti].mean() and edit_dist1_arr[mdti].mean()>edit_dist2_arr[mdti].mean() and edit_dist2_arr[mdti].mean()>edit_dist_unrel_arr[mdti].mean()

    # edit_distance_activities_at_decision_time = {dist: {word:edit_distance_activities[dist][word][decision_times[word_idx],:] for word_idx,word in enumerate(vocab)}  for dist in edit_distance_activities } 
    
    # fig,ax = plt.subplots()
    # analysis.plot_lexical_competition_data(fig, ax, t, edit_distance_activities)
    # ax.vlines(mdt,-1,1)
    # ax.set_ylim([0,1])

    # filtered_decision_times = {dist: {stim_word:t[decision_indices[vocab.index(stim_word)]] for stim_word,activities in edit_distance_activities[dist].items() if decision_indices[vocab.index(stim_word)] != -1 and activities.shape[1]>0} for dist in edit_distance_activities.keys()}
    # mdt = np.mean([vi for vo in filtered_decision_times.values() for vi in vo.values()])



    # plt.show()
    # plt.close(fig)
    # assert False
    # return 1,1,1,1,1

    if not np.any(filter_mask):
        print("------------- no decision was made --------------------")
        return -1,-1,-1,-1, int(np.sum(filter_mask)) #acc, n_decisions, correct_word_sup, correct_edit_dist, n_unstable
    if False:#np.any(filter_mask):
        # save_fp = os.path.join(".",sims_dir_name)
        # if not os.path.exists(save_fp):
        #     os.makedirs(save_fp)
    
        # mm = M_method
        # xm = param_dict["xZero_args"]["method"]
        # ft = param_dict["integrator_args"]["forcing_type"]
        # info = f"M{mm}_X{xm}_IF{ft}_A{'na'}"
        # current_dirs = [int(name.split("-")[-1]) for name in os.listdir(save_fp)]
        # if len(current_dirs)>0:
        #     new_dir = os.path.join(save_fp, info+"-sim-"+str(max(current_dirs)+1))
        # else:
        #     new_dir = os.path.join(save_fp, info+"-sim-"+str(0))
        # os.makedirs(new_dir)
        # print("all input words produced unstable activity trajectories; current run aborted")
        # with open(os.path.join(new_dir,"metadata.txt"),"w") as f:
        #     f.write(f"Accuracy data:\n\tNA\n\n")
        #     f.write(f"Model parameters:\n\t")
        #     for k,v in param_dict.items():
        #         f.write(f"{str(k)} - \n\t{v}\n\n")
        for i in range(xHistories.shape[-1]):
            nontgt_wd_mask = np.array([1 if j>(model_size-len(vocab)) or j==i else 0 for j in range(model_size)])
            # fig,ax=plt.subplots(112)
            plt.imshow(xHistories[:,:,i])
            # ax.
            # plt.savefig(os.path.join(new_dir,f"imshow_input_{vocab[i]}_wd_activites.png"))
            plt.show()
            # plt.plot(xHistories[:,nontgt_wd_mask,i],color="grey") #other words
            # plt.plot(xHistories[:,i+(model_size-len(vocab)),i],color="red") #input word
            # plt.savefig(os.path.join(new_dir,f"input_{vocab[i]}_wd_activites.png"))
            plt.close()
        return


    xHistories_d = xHistories[:,:,filter_mask]
    vocab = [w for i,w in enumerate(vocab) if filter_mask[i]]
    # median_activation = np.median(xHistories)
    # filter_mask = np.ones(xHistories.shape[-1]).astype(bool)
    # for i in range(xHistories.shape[-1]):
    #     if np.any(xHistories[:,:,i]>max_act_factor):
    #         filter_mask[i]=False
    # xHistories =

    # decisions,correct_decisions,decision_times = analysis.get_accuracy_end(model_params, t, vocab, pct=pct, xHistories=xHistories, compute_xHistories=False)
    decisions,decision_indices,correct_decisions,decision_times = analysis.get_accuracy_flexible(model_params, t, vocab, pct=99, xHistories=xHistories_d, compute_xHistories=False)
    # print(correct_decisions)

    # print("---------decision, correct decisions, decision times---------")
    # print(decisions)
    # print(correct_decisions)
    # print(decision_times)
    num_correct = int(np.sum(correct_decisions))
    vocab_len = len(decisions)
    num_leftout = int(np.sum(decisions<0)) #since decisions[i]=-1 if no decision was made

    # num_correct,num_leftout,vocab_len = analysis.get_accuracy_deterministic(model_params, t, accumulated_thresh, flexible=False)

    correct_decisions = correct_decisions.astype(bool)
    # print(correct_decisions)
    vocab_correct = [ w for i,w in enumerate(vocab) if correct_decisions[i]  and filter_mask[i]]
    xHistories_correct = xHistories_d[:,:,correct_decisions]






    # ----------------------------------------------- word superiority -----------------------------------------------

    psuedoword_list = analysis.make_psuedoword_list(model_params.letterlist,vocab,n_pseudowords=None, rng=None)
    pw_xHistories, pw_filter_mask = analysis.get_xHistories(model_params, t, psuedoword_list, filter_condition=analysis.filter_high_activations_by_median, filter_kwargs=filter_kwargs)

    # word_xHistory = integrate(model, wordlist, times) #these will also be used later to compute lexical competition results
    # psuedoword_xHistory = integrate(model, psuedoword_list, times)




    # print(xHistories.shape, "ppppppppppppp")
    xHistories = xHistories.transpose(0,2,1)
    pw_xHistories = pw_xHistories.transpose(0,2,1)
    word_decision_indices = analysis.compute_cumulative_gap_decision_indices(xHistories, t, stop_thresh=.05, time_axis = 0, node_axis=-1, keepdims=False)
    psuedoword_decision_indices = analysis.compute_cumulative_gap_decision_indices(pw_xHistories, t, stop_thresh=.05, time_axis = 0, node_axis=-1, keepdims=False)

    print(xHistories.shape, len(model_params.letterlist), len(model_params.wordlist), len(vocab))

    w_decision_mask = word_decision_indices!=-1
    pw_decision_mask = word_decision_indices!=-1

    if True:#np.any(w_decision_mask) and np.any(pw_decision_mask):
        mean_decision_index = round(
                                    int( 
                                        (word_decision_indices.sum()+psuedoword_decision_indices.sum())
                                        /(len(word_decision_indices.reshape(-1))+len(psuedoword_decision_indices.reshape(-1)))  
                                        )
                                    )



        word_letter_activations = np.vstack([
                                             
                                            xHistories[:,w_idx,model_params.nodelist.index(letter)] 
                                            for w_idx,word in enumerate(vocab)# if w_decision_mask[w_idx]
                                            for letter in analysis.get_positional_letters(word)
                                            ]).T
        # plt.imshow(word_letter_activations)
        # plt.show()
        print(np.mean(word_letter_activations[0,:]==0.9))
        ps_word_letter_activations = np.vstack([
                                                pw_xHistories[:,w_idx,model_params.nodelist.index(letter)] 
                                                for w_idx,word in enumerate(psuedoword_list) #if pw_decision_mask[w_idx]
                                                for letter in analysis.get_positional_letters(word) 
                                                ]).T

        # fig_ws,ax_ws = plt.subplots()
        # ax_ws.plot(word_letter_activations.mean(axis=1), c='r')
        # ax_ws.plot(ps_word_letter_activations.mean(axis=1), c='black')
        # plt.show()



        print(word_letter_activations.shape, "llllllllllllllllllllllll")
        print(mean_decision_index)
        correct_word_sup = word_letter_activations[mean_decision_index,:].mean() >  ps_word_letter_activations[mean_decision_index,:].mean()
    else:
        correct_word_sup = -1









    # fig1,ax1=plt.subplots()
    # ax1.set_title(f"Word Sup: acc={num_correct}/{vocab_len-num_leftout} n_words:{vocab_len}")
    # analysis.make_word_superiority_plot(ax1, model_params, vocab, t)
    # analysis.make_word_superiority_plot(ax1, model_params, vocab, t, xHistories_words=xHistories, xHistories_nonwords=None, pct=pct,
    #                                                                                 filter_condition=analysis.filter_high_activations_by_median, filter_kwargs=filter_kwargs) #all xHistories in this case - all input words should be compared with all input pseudowords
    # correct_word_sup = analysis.get_word_superiority_acc(model_params, vocab, t, xHistories_words=xHistories, xHistories_nonwords=None, pct=pct, rng=rng,
    #                                                                                 filter_condition=analysis.filter_high_activations_by_median, filter_kwargs=filter_kwargs) #all xHistories in this case - all input words should be compared with all input pseudowords




    # fig2,ax2=plt.subplots()
    # ax2.set_title(f"Edit dist plot: acc={num_correct}/{vocab_len-num_leftout} n_words:{vocab_len}")
    # # analysis.make_substitution_plot(ax2, t, model_params, vocab,accumulated_thresh, flexible=False)
    # analysis.make_substitution_plot(ax2, t, model_params, vocab_correct, xHistories=xHistories_correct, pct=pct)
    # plt.show()
    # plt.close(fig2)
    if correct_decisions.sum()!=0:
        correct_edit_dist = correct_edit_dist #analysis.get_substitution_acc(t, model_params, vocab_correct, xHistories=xHistories_correct, pct=pct)
    else:
        correct_edit_dist = -1




    #info-sim-num
    # print(round(1.23432,4))
    acc=round(num_correct/(vocab_len-num_leftout),3) if vocab_len!=num_leftout else 0.0
    # acc=".".join([v if i==0 else v[:3] for i,v, in enumerate(str(acc).split("."))])


    #------------------setting up dirs well save to----------------------
    # save_fp = os.path.join(".",sims_dir_name)
    # if not os.path.exists(save_fp):
    #     os.makedirs(save_fp)

    # mm = M_method
    # xm = param_dict["xZero_args"]["method"]
    # ft = param_dict["integrator_args"]["forcing_type"]
    # info = f"M{mm}_X{xm}_IF{ft}_A{acc}"
    # current_dirs = [int(name.split("-")[-1]) for name in os.listdir(save_fp)]
    # if len(current_dirs)>0:
    #     new_dir = os.path.join(save_fp, info+"-sim-"+str(max(current_dirs)+1))
    # else:
    #     new_dir = os.path.join(save_fp, info+"-sim-"+str(0))
    # os.makedirs(new_dir)
    #------------------------------------------------------------------


    # fig1.savefig(os.path.join(new_dir,"word_superiority.png"))
    # fig2.savefig(os.path.join(new_dir,"edit_distance_plot.png"))
    # plt.close()
    # with open(os.path.join(new_dir,"metadata.txt"),"w") as f:
    #     f.write(f"Accuracy data:\n\tnum_correct: {num_correct} ; num_leftout: {num_leftout} ; n_words:{vocab_len}\n\n")
    #     f.write(f"Model parameters:\n\t")
    #     for k,v in param_dict.items():
    #         f.write(f"{str(k)} - \n\t{v}\n\n")


    return acc, int(vocab_len-num_leftout), int(correct_word_sup), int(correct_edit_dist), int(np.sum(filter_mask))   ##acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable  #(n_stabile is the number that had activity mostly bounded, i.e. no nodes greater than x*99th_percentile(activities_over_time))




















def ode_integrate(model_params, t, vocab):
    xHistories = model_params.integrator(vocab)
    return xHistories








def full_output_for_params(vocab,param_dict,times, pct=1, rng=None, filter_kwargs={"high_activity_factor":100000}): #high_activity_factor for filtering out trails where activities are unstable or get really large
    pylab_pretty_plot()

    if rng is None:
        rng = np.random.default_rng()
    # rng_voc = np.random.default_rng()
    voc_size = param_dict["vocab_size"]
    vocab = rng.choice(vocab,size=voc_size,replace=False)

    model_params = glv.model_parameters.Parameters(vocab)
    sims_dir_name = "method_param_sims10000"


    model_params.set_pattern_generator(**param_dict["pattern_args"])
    if param_dict["r_args"]["wgrowth"] is None: #forcing wgrowth to be the same as lgrowth for None
        param_dict["r_args"]["wgrowth"] = param_dict["r_args"]["lgrowth"]
    r = model_params.set_r(**param_dict["r_args"])
    # print(param_dict)
    try:
        M_method = param_dict["M_args"].pop("method")
    except:
        print(param_dict["M_args"])
        raise

    if M_method =="m1":
        M = model_params.set_M_method1()
    elif M_method =="m2":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        M = model_params.set_M_method2( **param_dict["M_args"] )

    elif M_method =="m3":
        M = model_params.set_M_method3( **param_dict["M_args"] )
    elif M_method =="m4":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        if "ltw" not in param_dict["M_args"].keys():
            param_dict["M_args"]["ltw"] = param_dict["M_args"]["wtl"]

        M = model_params.set_M_method4( **param_dict["M_args"] )
    elif M_method =="m5":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]

        M = model_params.set_M_method5( **param_dict["M_args"] )
    else:
        # mmethod=param_dict["M_args"]["method"]
        raise Exception(f"parameters provided for 'set_M' should be chosen from ['m1', 'm2' ,'m3', 'm4'] (not '{M_method}')")


    eigenvalues, eigenvectors = np.linalg.eig((M + M.T)/2)
    print(eigenvalues.min(), eigenvalues.max(), eigenvalues.mean(), np.median(eigenvalues))
    maxeigval = eigenvalues.real.max()
    print( np.all(M==M.T))
    # if eigenvalues.real.max()>3:
    #     return (-2,)*5

    # plt.imshow(M)
    # plt.show()
    model_params.set_xZero_generator(**param_dict["xZero_args"])
    if param_dict["integrator_args"]["forcing_type"] != "n":
        model_params.set_forcing_generator(**param_dict["forcing_args"])

    # times = np.linspace(0,40,200)

    integrator = model_params.set_integrator(times,forcing_type=param_dict["integrator_args"]["forcing_type"])


    # min_activation, max_activation = 0, 100
    # # stop_thresh = "0pct"
    # stop_thresh = "min"


    # model_size=len(model_params.nodelist)
    # # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # # accumulated_thresh=0.0001/np.log(len(model_params.nodelist))#/np.log(80)

    # # vocab = model_params.wordlist



    # # we'll use the xHistory computed for the full vocab for both WS and LC

    # # ----- integrate for word-stimuli
    # w_xHistories = model_params.batch_integrate_(times, vocab).transpose(0,2,1)
    # w_letter_layer_activities = w_xHistories[:,:, :len(model_params.letterlist)]
    # w_word_layer_activities = w_xHistories[:,:, -len(model_params.wordlist):]

    # w_decision_indices = analysis.compute_cumulative_gap_decision_indices(w_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)
    # # print(times.shape, w_decision_indices.shape)

    # w_decision_times = np.take_along_axis(times[ : , None, None ] , w_decision_indices, axis=0) # times: (n_times,1,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    # w_decision_activations = np.take_along_axis(w_xHistories, w_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    # w_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    # w_decision_mask = w_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    # w_correct_decision_mask = [w_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]
    # # print(w_predicted_nodes)
    # # print()
    # w_in_bounds_mask = np.all( (w_xHistories >= min_activation) & (w_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word

    # # print(w_in_bounds_mask.sum())
    # # # plt.plot(times, w_xHistories[:,i,:], c='green', alpha=0.6)
    # # plt.plot(times, w_word_layer_activities[:,0,:], c='black')
    # # plt.plot(times, w_word_layer_activities[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # # if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
    # #     plt.vlines( times[int(round(w_decision_indices.squeeze()[0]))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
    # # plt.xlim([0,20])
    # # plt.ylim([0,.01])
    # # plt.show()




    # times = np.linspace(0,5,100)
    # integrator = model_params.set_integrator(times,forcing_type=param_dict["integrator_args"]["forcing_type"])


    min_activation, max_activation = 0, 100
    # stop_thresh = "min"
    stop_threshs = {50:0.002, 100:0.001, 1000:0.00001}
    # stop_threshs = {50:0.005, 100:0.001, 1000:0.00002}

    # # stop_thresh = 0.0001
    stop_thresh = stop_threshs[voc_size]
    # stop_thresh = .05/voc_size

    model_size=len(model_params.nodelist)
    # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # accumulated_thresh=0.0001/np.log(len(model_params.nodelist))#/np.log(80)

    # vocab = model_params.wordlist



    # we'll use the xHistory computed for the full vocab for both WS and LC

    # ----- integrate for word-stimuli
    # w_xHistories_ia = model_params.batch_integrate_IA(times, vocab).transpose(0,2,1)
    # w_letter_layer_activities_ia = w_xHistories_ia[:,:, :len(model_params.letterlist)]
    # w_word_layer_activities_ia = w_xHistories_ia[:,:, -len(model_params.wordlist):]

    w_xHistories = model_params.batch_integrate(times, vocab).transpose(0,2,1)
    w_letter_layer_activities = w_xHistories[:,:, :len(model_params.letterlist)]
    w_word_layer_activities = w_xHistories[:,:, -len(model_params.wordlist):]


    # # ----------- if we want to look at solutions to the SDE moment equations and Euler Maruyama approximation

    # w_xMeans_sde, w_xVars_sde = analysis.sde_integrate(model_params, times, vocab[:1], sigma=2, order = 2)
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=2, n_sims=1000, step_size=10000)

    # # plt.plot(times, w_xHistories[:,0,model_params.nodelist.index(vocab[0])], c='black')
    # # plt.plot(times, w_xMeans_sde[:,0,model_params.nodelist.index(vocab[0])], c='b')
    # # plt.plot(times, w_xMeans_em[:,0,model_params.nodelist.index(vocab[0])], c='r')
    # # plt.ylim([0,4])
    # # plt.show()


    # fig,ax=plt.subplots()
    # for wi in range(10):
    #     # ax.fill_between(times, w_xMeans_sde[:,0,model_params.nodelist.index(vocab[0])] - np.einsum("ijkk->ijk",w_xVars_sde)[:,0,model_params.nodelist.index(vocab[0])], 
    #     #                        w_xMeans_sde[:,0,model_params.nodelist.index(vocab[0])] + np.einsum("ijkk->ijk",w_xVars_sde)[:,0,model_params.nodelist.index(vocab[0])], 
    #     #                        alpha=0.1, interpolate=True, color="r"
    #     #                 )
    #     # ax.plot(times, w_xMeans_sde[:,0,model_params.nodelist.index(vocab[0])], c="r", label = 'Moment equation')

    #     ax.fill_between(times, w_xMeans_em[:,0,model_params.nodelist.index(vocab[wi])] - np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], 
    #                            w_xMeans_em[:,0,model_params.nodelist.index(vocab[wi])] + np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], 
    #                            alpha=0.1, interpolate=True, color="b"
    #                     )
    #     ax.plot(times, w_xMeans_em[:,0,model_params.nodelist.index(vocab[wi])], c="b", label = 'Euler-Maruyama')

    #     plt.plot(times, w_xHistories[:,0,model_params.nodelist.index(vocab[wi])], c='black', label = "Deterministic")


    # # ax.legend(loc="upper right")
    # plt.savefig(r"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\SDE_moment_closure_vs_EM-medianofbest_s1.1_trials1_steps10000000.png", dpi=1000, transparent=True)
    # plt.show()
    # #------------------------^^^^^^^^^^^^^^^^^^^^^--------------------------------------









    # plt.plot(times, np.einsum("ijkk->ijk",w_xVars_sde)[:,0,model_params.nodelist.index(vocab[0])], c='b')


    # plt.plot(times, np.einsum("ijkk->ijk",w_xVars_sde)[:,0,model_params.nodelist.index(vocab[0])], c='b')

    # # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='r')
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=1000)
    # # indices = np.arange(0,len(times_em),len(times_em//100))
    # # w_xMeans_em, w_xVars_em, times_em = w_xMeans_em[indices], w_xVars_em[indices], times_em[indices]
    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='orange')
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=10000)
    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='yellow')
    # # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=1000, step_size=100000)
    # # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='grey')
    # plt.show()


    w_decision_indices = analysis.compute_cumulative_gap_decision_indices(w_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)
    # print(times.shape, w_decision_indices.shape)

    w_decision_times = np.take_along_axis(times[ : , None, None ] , w_decision_indices, axis=0) # times: (n_times,1,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    w_decision_activations = np.take_along_axis(w_xHistories, w_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    w_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    w_decision_mask = w_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    w_correct_decision_mask = [w_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]
    # print(w_predicted_nodes)
    # print()
    w_in_bounds_mask = np.all( (w_xHistories >= min_activation) & (w_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word

    # fig,axes = plt.subplots(1,2)
    # axes[0].plot(times, w_word_layer_activities[:,0,:], c='black')
    # axes[0].plot(times, w_word_layer_activities[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
    #     axes[0].vlines( times[int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
    # axes[0].set_xlim([0,20])
    # axes[0].set_ylim([-.5,2])

    # axes[1].plot(times, w_word_layer_activities_ia[:,0,:], c='black')
    # axes[1].plot(times, w_word_layer_activities_ia[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
    #     axes[1].vlines( times[int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
    # axes[1].set_xlim([0,20])
    # axes[1].set_ylim([-.5,2])
    # plt.show()





    # ----- integrate for nonword-stimuli
    nonword_list = analysis.make_psuedoword_list(model_params.letterlist,vocab,n_pseudowords=None, rng=None) # matches the empirical distribution of lengths, and pulls letters from those that the model already has nodes for

    nw_xHistories = model_params.batch_integrate(times, nonword_list).transpose(0,2,1)
    nw_letter_layer_activities = nw_xHistories[:,:, :len(model_params.letterlist)]
    nw_word_layer_activities = nw_xHistories[:,:, -len(model_params.wordlist):]

    nw_decision_indices = analysis.compute_cumulative_gap_decision_indices(nw_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)

    nw_decision_times = np.take_along_axis(times[ : , None, None] , nw_decision_indices, axis=0) # times: (1,n_times,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    nw_decision_activations = np.take_along_axis(nw_xHistories, nw_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    # nw_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    nw_decision_mask = nw_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    # nw_correct_decision_mask = [nw_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]
    nw_in_bounds_mask = np.all( (nw_xHistories >= min_activation) & (nw_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word



    # plt.plot(times, nw_word_layer_activities[:,0,:], c='black')
    # # plt.plot(times, w_word_layer_activities[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # if not np.all( nw_decision_indices.squeeze()[nw_decision_mask] ==-1 ):
    #     plt.vlines( times[int(round(nw_decision_indices.squeeze()[nw_decision_mask].mean()))] , min_activation, min([max_activation, nw_word_layer_activities.max()]) , color='green')
    # plt.xlim([0,20])
    # plt.ylim([0,.01])
    # plt.show()



    if (not np.any(w_decision_mask & w_in_bounds_mask)) or (not np.any(nw_decision_mask & nw_in_bounds_mask)):
        # for i in range(len(vocab)):
        #     if True:#not w_decision_mask[i] or not w_in_bounds_mask[i]:
        #         plt.plot(times, w_xHistories[:,i,:], c='green', alpha=0.6)
        #         plt.plot(times, w_word_layer_activities[:,i,:], c='black')
        #         plt.plot(times, w_word_layer_activities[:,i,model_params.wordlist.index(vocab[0])], c='r')
        #         if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
        #             plt.vlines( times[int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
        #         plt.xlim([0,20])
        #         plt.title(vocab[i] + str(w_decision_mask[i])+str(w_in_bounds_mask[i]))
        #         plt.show()
        #         break
        accuracy = -1

        # ----- accuracy ----
        w_valid_trial_mask  = w_decision_mask  & w_in_bounds_mask 
        w_correct_trial_mask = w_decision_mask & w_correct_decision_mask & w_in_bounds_mask

        num_correct = int(np.sum(w_correct_trial_mask))
        valid_trials = np.sum(w_valid_trial_mask)

        accuracy = num_correct / valid_trials

        num_in_bounds = w_in_bounds_mask.sum()
        num_decisions = (w_decision_mask&w_in_bounds_mask).sum() #we're only really interested in how many decisions were made of the in-bounds trials

        return accuracy, int(num_decisions), -1, -1, int(num_in_bounds), maxeigval  ##acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable  #(n_stabile is the number that had activity mostly bounded, i.e. no nodes greater than x*99th_percentile(activities_over_time))

        # return (-1,)*6  #indicating invalid results for all return values


    # ----------------------------------------------- word superiority -----------------------------------------------

    # letters_mask = np.zeros(( len(vocab), len(model_params.letterlist) )) #(n_stimulus_words, n_letter_nodes) 
    w_letters_mask = \
        np.array( [
                    [letter in analysis.get_positional_letters(word) for i,letter in enumerate(model_params.letterlist)] 
                        for j,word in enumerate(vocab)
                  ] )
    nw_letters_mask = \
        np.array( [
                    [letter in analysis.get_positional_letters(word) for i,letter in enumerate(model_params.letterlist)] 
                        for j,word in enumerate(nonword_list)
                  ] )

    # stimulus words/nonwords that resulted in in-bounds activity and where decision was made (cumulative gap passed threshold within simulation time window)
    w_valid_trial_mask  = w_decision_mask  & w_in_bounds_mask  # (n_stimulus_words,)
    nw_valid_trial_mask = nw_decision_mask & nw_in_bounds_mask

    # print(w_decision_mask.shape, w_in_bounds_mask.shape)
    # print(w_letters_mask.shape, w_valid_trial_mask.shape, w_letter_layer_activities.shape)

    w_letter_activations = w_letter_layer_activities[:, w_letters_mask & w_valid_trial_mask[:, None] ]
    nw_letter_activations = nw_letter_layer_activities[:, nw_letters_mask & nw_valid_trial_mask[:, None] ]

    WS_mean_decision_index = int(np.hstack([w_decision_indices.squeeze()[w_decision_mask], nw_decision_indices.squeeze()[nw_decision_mask]]).mean())
    print(WS_mean_decision_index)
    correct_WS = w_letter_activations[WS_mean_decision_index,:].mean() > nw_letter_activations[WS_mean_decision_index,:].mean()


    import scipy.stats as sps
    # -------------- plot WS ----------------
    fig,ax = plt.subplots()
    w_letter_advantage = w_letter_activations - nw_letter_activations.mean(axis=1,keepdims=True)
    nw_letter_advantage = nw_letter_activations - nw_letter_activations.mean(axis=1,keepdims=True)
    ax.fill_between(times, w_letter_advantage.mean(axis=1) + sps.sem(w_letter_advantage, axis=1)*sps.norm.ppf(.025), 
                           w_letter_advantage.mean(axis=1) + sps.sem(w_letter_advantage, axis=1)*sps.norm.ppf(.975), 
                           alpha=0.5, interpolate=True, color="r"
                    )
    ax.plot(times, w_letter_advantage.mean(axis=1), c="r", label = 'words')

    ax.fill_between(times, nw_letter_advantage.mean(axis=1) + sps.sem(nw_letter_advantage, axis=1)*sps.norm.ppf(.025), 
                           nw_letter_advantage.mean(axis=1) + sps.sem(nw_letter_advantage, axis=1)*sps.norm.ppf(.975), 
                           alpha=0.5, interpolate=True, color="black"
                )
    ax.plot(times, nw_letter_advantage.mean(axis=1), c="black", linestyle = 'dashed', label = 'non-words')

    ax.legend(loc="upper right")
    plt.savefig(rf"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\WS_example_{M_method}_vocsize-{voc_size}_thresh-{stop_thresh}.png", dpi=1000, transparent=True)
    plt.clf()

    # plt.show()


    # ------------------------------------------------ lexical competition --------------------------------------------------

    # for LC, we want to filter out trials where: 1. no decision was made, 2. the incorrect decision was made, and 3. activity in at least one node exited an arbitrary "in-bounds" region (e.g. [0,10])
    # --- each of these should be a boolean mask with shape (n_stimulus_words,)

    pairwise_edit_distance=analysis.get_pairwise_edit_distance(vocab, model_params.wordlist) # (n_stimulus_words, n_word_nodes)
    edit_dist0_mask = pairwise_edit_distance == 0
    edit_dist1_mask = pairwise_edit_distance == 1
    edit_dist2_mask = pairwise_edit_distance == 2
    edit_distUR_mask = pairwise_edit_distance  > 2


    w_correct_trial_mask = w_decision_mask & w_correct_decision_mask & w_in_bounds_mask

    # the 2D boolean mask along the last two dimensions of word_layer_activities results in a (n_times, n_true) shaped array
    edit_dist0_activities = w_word_layer_activities[:, edit_dist0_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)
    edit_dist1_activities = w_word_layer_activities[:, edit_dist1_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)
    edit_dist2_activities = w_word_layer_activities[:, edit_dist2_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)
    edit_distUR_activities = w_word_layer_activities[:, edit_distUR_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)

    # if we lack one of the edit-distance neighborhoods, edit_distance trail is invalid
    if any([min(edit_dist0_activities.shape)==0, min(edit_dist1_activities.shape)==0, min(edit_dist2_activities.shape)==0, min(edit_distUR_activities.shape)==0,]):
        correct_LC = -1
    else:
        LC_mean_decision_index = int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))
        correct_LC = edit_dist0_activities[LC_mean_decision_index, :].mean() > edit_dist1_activities[LC_mean_decision_index, :].mean() > edit_dist2_activities[LC_mean_decision_index, :].mean() > edit_distUR_activities[LC_mean_decision_index, :].mean()

        # plt.plot(edit_dist0_activities.mean(axis=1), c='red')
        # plt.plot(edit_dist1_activities.mean(axis=1), c='orange')
        # plt.plot(edit_dist2_activities.mean(axis=1), c='yellow')
        # plt.plot(edit_distUR_activities.mean(axis=1), c='grey')
        # plt.show()



    # ------------- plot LC ---------------
    import matplotlib as mpl

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f"truncated({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap

    cmap = truncate_colormap(mpl.colormaps["plasma"], minval=0.0, maxval=0.66, n=256)
    norm = mpl.colors.Normalize(vmin=0,vmax=1)
    fig,ax = plt.subplots()
    # colors = ["#ff0021", "#ff00ca", "#c500ff", "#4500ff"]
    colors = [cmap(norm(1)), cmap(norm(0.66)), cmap(norm(0.33)), cmap(norm(0.0)) ]
    for data, color, linestyle, label in [(edit_dist0_activities,colors[0], "solid", "stimulus"), 
                                (edit_dist1_activities,colors[1], "dashdot", "1-step"), 
                                (edit_dist2_activities,colors[2],"dashed", "2-step"), 
                                (edit_distUR_activities,colors[3],"dotted", ">2-step")]:
        # ax.fill_between(t, np.percentile(data, 16, axis=0), np.percentile(data, 100-16, axis=0), alpha=0.1, interpolate=True, color=color)
        ax.fill_between(times, data.mean(axis=1) + sps.sem(data, axis=1)*sps.norm.ppf(.025), 
                           data.mean(axis=1) + sps.sem(data, axis=1)*sps.norm.ppf(.975), 
                           alpha=0.5, interpolate=True, color=color)
        # print(data[0,0],color)
        ax.plot(times, data.mean(axis=1), color = color, linestyle=linestyle, label=label)
        # ax.plot(t, np.percentile(data, 35, axis=0), color=color)
        # ax.plot(t, np.percentile(data, 65, axis=0), color=color)
    # ax.errorbar(t,np.mean(np.vstack(input_word_activity),axis=0),       yerr=sst.sem(np.vstack(input_word_activity),axis=0),       color='red',   ecolor='red')

    ax.legend(loc="upper right")
    plt.savefig(rf"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\LC_example_{M_method}_vocsize-{voc_size}_thresh-{stop_thresh}.png", dpi=1000, transparent=True)
    plt.clf()
    # plt.show()








    # ----- accuracy ----
    num_correct = int(np.sum(w_correct_trial_mask))
    valid_trials = np.sum(w_valid_trial_mask)

    accuracy = num_correct / valid_trials

    num_in_bounds = w_in_bounds_mask.sum()
    num_decisions = (w_decision_mask&w_in_bounds_mask).sum() #we're only really interested in how many decisions were made of the in-bounds trials

    return accuracy, int(num_decisions), int(correct_WS), int(correct_LC), int(num_in_bounds), maxeigval  ##acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable  #(n_stabile is the number that had activity mostly bounded, i.e. no nodes greater than x*99th_percentile(activities_over_time))






























# version used for a slightly more realistic WS setup (masking out target letters)
def full_output_for_params_(vocab,param_dict,times, pct=1, rng=None, filter_kwargs={"high_activity_factor":100000}): #high_activity_factor for filtering out trails where activities are unstable or get really large
    if rng is None:
        rng = np.random.default_rng()
    # rng_voc = np.random.default_rng()
    voc_size = param_dict["vocab_size"]
    vocab = rng.choice(vocab,size=voc_size,replace=False)

    model_params = glv.model_parameters.Parameters(vocab)
    sims_dir_name = "method_param_sims10000"


    model_params.set_pattern_generator(**param_dict["pattern_args"])
    if param_dict["r_args"]["wgrowth"] is None: #forcing wgrowth to be the same as lgrowth for None
        param_dict["r_args"]["wgrowth"] = param_dict["r_args"]["lgrowth"]
    r = model_params.set_r(**param_dict["r_args"])
    # print(param_dict)
    try:
        M_method = param_dict["M_args"].pop("method")
    except:
        print(param_dict["M_args"])
        raise

    if M_method =="m1":
        M = model_params.set_M_method1()
    elif M_method =="m2":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        M = model_params.set_M_method2( **param_dict["M_args"] )

    elif M_method =="m3":
        M = model_params.set_M_method3( **param_dict["M_args"] )
    elif M_method =="m4":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        if "ltw" not in param_dict["M_args"].keys():
            param_dict["M_args"]["ltw"] = param_dict["M_args"]["wtl"]

        M = model_params.set_M_method4( **param_dict["M_args"] )
    elif M_method =="m5":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]

        M = model_params.set_M_method5( **param_dict["M_args"] )
    else:
        # mmethod=param_dict["M_args"]["method"]
        raise Exception(f"parameters provided for 'set_M' should be chosen from ['m1', 'm2' ,'m3', 'm4'] (not '{M_method}')")


    eigenvalues, eigenvectors = np.linalg.eig((M + M.T)/2)
    print(eigenvalues.min(), eigenvalues.max(), eigenvalues.mean(), np.median(eigenvalues))
    maxeigval = eigenvalues.real.max()
    print( np.all(M==M.T))
    # if eigenvalues.real.max()>3:
    #     return (-2,)*5

    # plt.imshow(M)
    # plt.show()
    model_params.set_xZero_generator(**param_dict["xZero_args"])
    if param_dict["integrator_args"]["forcing_type"] != "n":
        model_params.set_forcing_generator(**param_dict["forcing_args"])

    # times = np.linspace(0,40,200)

    integrator = model_params.set_integrator(times,forcing_type=param_dict["integrator_args"]["forcing_type"])


    # min_activation, max_activation = 0, 100
    # # stop_thresh = "0pct"
    # stop_thresh = "min"


    # model_size=len(model_params.nodelist)
    # # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # # accumulated_thresh=0.0001/np.log(len(model_params.nodelist))#/np.log(80)

    # # vocab = model_params.wordlist



    # # we'll use the xHistory computed for the full vocab for both WS and LC

    # # ----- integrate for word-stimuli
    # w_xHistories = model_params.batch_integrate_(times, vocab).transpose(0,2,1)
    # w_letter_layer_activities = w_xHistories[:,:, :len(model_params.letterlist)]
    # w_word_layer_activities = w_xHistories[:,:, -len(model_params.wordlist):]

    # w_decision_indices = analysis.compute_cumulative_gap_decision_indices(w_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)
    # # print(times.shape, w_decision_indices.shape)

    # w_decision_times = np.take_along_axis(times[ : , None, None ] , w_decision_indices, axis=0) # times: (n_times,1,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    # w_decision_activations = np.take_along_axis(w_xHistories, w_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    # w_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    # w_decision_mask = w_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    # w_correct_decision_mask = [w_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]
    # # print(w_predicted_nodes)
    # # print()
    # w_in_bounds_mask = np.all( (w_xHistories >= min_activation) & (w_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word

    # # print(w_in_bounds_mask.sum())
    # # # plt.plot(times, w_xHistories[:,i,:], c='green', alpha=0.6)
    # # plt.plot(times, w_word_layer_activities[:,0,:], c='black')
    # # plt.plot(times, w_word_layer_activities[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # # if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
    # #     plt.vlines( times[int(round(w_decision_indices.squeeze()[0]))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
    # # plt.xlim([0,20])
    # # plt.ylim([0,.01])
    # # plt.show()




    # times = np.linspace(0,200,2000)
    # integrator = model_params.set_integrator(times,forcing_type=param_dict["integrator_args"]["forcing_type"])


    min_activation, max_activation = 0, 100
    # stop_thresh = "min"
    stop_thresh = 0.0001


    model_size=len(model_params.nodelist)
    # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # accumulated_thresh=0.0001/np.log(len(model_params.nodelist))#/np.log(80)

    # vocab = model_params.wordlist



    # we'll use the xHistory computed for the full vocab for both WS and LC

    # ----- integrate for word-stimuli
    # w_xHistories_ia = model_params.batch_integrate_IA(times, vocab).transpose(0,2,1)
    # w_letter_layer_activities_ia = w_xHistories_ia[:,:, :len(model_params.letterlist)]
    # w_word_layer_activities_ia = w_xHistories_ia[:,:, -len(model_params.wordlist):]

    w_xHistories = model_params.batch_integrate(times, vocab).transpose(0,2,1)
    w_letter_layer_activities = w_xHistories[:,:, :len(model_params.letterlist)]
    w_word_layer_activities = w_xHistories[:,:, -len(model_params.wordlist):]


    # ----------- if we want to look at solutions to the SDE moment equations and Euler Maruyama approximation

    # w_xMeans_sde, w_xVars_sde = analysis.sde_integrate(model_params, times, vocab[:1], sigma=.5, order = 2)
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=100)


    # # plt.plot(times, w_xMeans_sde[:,0,model_params.nodelist.index(vocab[0])], c='b')
    # # plt.plot(times, w_xMeans_em[:,0,model_params.nodelist.index(vocab[0])], c='r')

    # plt.plot(times, np.einsum("ijkk->ijk",w_xVars_sde)[:,0,model_params.nodelist.index(vocab[0])], c='b')

    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='r')
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=1000)
    # # indices = np.arange(0,len(times_em),len(times_em//100))
    # # w_xMeans_em, w_xVars_em, times_em = w_xMeans_em[indices], w_xVars_em[indices], times_em[indices]
    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='orange')
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=10000)
    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='yellow')
    # # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=1000, step_size=100000)
    # # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='grey')
    # plt.show()


    w_decision_indices = analysis.compute_cumulative_gap_decision_indices(w_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)
    # print(times.shape, w_decision_indices.shape)

    w_decision_times = np.take_along_axis(times[ : , None, None ] , w_decision_indices, axis=0) # times: (n_times,1,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    w_decision_activations = np.take_along_axis(w_xHistories, w_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    w_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    w_decision_mask = w_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    w_correct_decision_mask = [w_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]
    # print(w_predicted_nodes)
    # print()
    w_in_bounds_mask = np.all( (w_xHistories >= min_activation) & (w_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word

    # fig,axes = plt.subplots(1,2)
    # axes[0].plot(times, w_word_layer_activities[:,0,:], c='black')
    # axes[0].plot(times, w_word_layer_activities[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
    #     axes[0].vlines( times[int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
    # axes[0].set_xlim([0,20])
    # axes[0].set_ylim([-.5,2])

    # axes[1].plot(times, w_word_layer_activities_ia[:,0,:], c='black')
    # axes[1].plot(times, w_word_layer_activities_ia[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
    #     axes[1].vlines( times[int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
    # axes[1].set_xlim([0,20])
    # axes[1].set_ylim([-.5,2])
    # plt.show()





    # ----- integrate for nonword-stimuli
    nonword_list = analysis.make_psuedoword_list(model_params.letterlist,vocab,n_pseudowords=None, rng=None) # matches the empirical distribution of lengths, and pulls letters from those that the model already has nodes for

    nw_xHistories = model_params.batch_integrate(times, nonword_list).transpose(0,2,1)
    nw_letter_layer_activities = nw_xHistories[:,:, :len(model_params.letterlist)]
    nw_word_layer_activities = nw_xHistories[:,:, -len(model_params.wordlist):]

    nw_decision_indices = analysis.compute_cumulative_gap_decision_indices(nw_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)

    nw_decision_times = np.take_along_axis(times[ : , None, None] , nw_decision_indices, axis=0) # times: (1,n_times,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    nw_decision_activations = np.take_along_axis(nw_xHistories, nw_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    # nw_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    nw_decision_mask = nw_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    # nw_correct_decision_mask = [nw_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]
    nw_in_bounds_mask = np.all( (nw_xHistories >= min_activation) & (nw_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word



    # plt.plot(times, nw_word_layer_activities[:,0,:], c='black')
    # # plt.plot(times, w_word_layer_activities[:,0,model_params.wordlist.index(vocab[0])], c='r')
    # if not np.all( nw_decision_indices.squeeze()[nw_decision_mask] ==-1 ):
    #     plt.vlines( times[int(round(nw_decision_indices.squeeze()[nw_decision_mask].mean()))] , min_activation, min([max_activation, nw_word_layer_activities.max()]) , color='green')
    # plt.xlim([0,20])
    # plt.ylim([0,.01])
    # plt.show()



    if (not np.any(w_decision_mask & w_in_bounds_mask)) or (not np.any(nw_decision_mask & nw_in_bounds_mask)):
        # for i in range(len(vocab)):
        #     if True:#not w_decision_mask[i] or not w_in_bounds_mask[i]:
        #         plt.plot(times, w_xHistories[:,i,:], c='green', alpha=0.6)
        #         plt.plot(times, w_word_layer_activities[:,i,:], c='black')
        #         plt.plot(times, w_word_layer_activities[:,i,model_params.wordlist.index(vocab[0])], c='r')
        #         if not np.all( w_decision_indices.squeeze()[w_decision_mask] ==-1 ):
        #             plt.vlines( times[int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))] , min_activation, min([max_activation, w_word_layer_activities.max()]) , color='green')
        #         plt.xlim([0,20])
        #         plt.title(vocab[i] + str(w_decision_mask[i])+str(w_in_bounds_mask[i]))
        #         plt.show()
        #         break
        accuracy = -1

        # ----- accuracy ----
        w_valid_trial_mask  = w_decision_mask  & w_in_bounds_mask 
        w_correct_trial_mask = w_decision_mask & w_correct_decision_mask & w_in_bounds_mask

        num_correct = int(np.sum(w_correct_trial_mask))
        valid_trials = np.sum(w_valid_trial_mask)

        accuracy = num_correct / valid_trials

        num_in_bounds = w_in_bounds_mask.sum()
        num_decisions = (w_decision_mask&w_in_bounds_mask).sum() #we're only really interested in how many decisions were made of the in-bounds trials

        return accuracy, int(num_decisions), -1, -1, int(num_in_bounds), maxeigval  ##acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable  #(n_stabile is the number that had activity mostly bounded, i.e. no nodes greater than x*99th_percentile(activities_over_time))

        # return (-1,)*6  #indicating invalid results for all return values


    # ----------------------------------------------- word superiority -----------------------------------------------

    # letters_mask = np.zeros(( len(vocab), len(model_params.letterlist) )) #(n_stimulus_words, n_letter_nodes) 
    w_letters_mask = \
        np.array( [
                    [letter in analysis.get_positional_letters(word) for i,letter in enumerate(model_params.letterlist)] 
                        for j,word in enumerate(vocab)
                  ] )
    nw_letters_mask = \
        np.array( [
                    [letter in analysis.get_positional_letters(word) for i,letter in enumerate(model_params.letterlist)] 
                        for j,word in enumerate(nonword_list)
                  ] )

    # stimulus words/nonwords that resulted in in-bounds activity and where decision was made (cumulative gap passed threshold within simulation time window)
    w_valid_trial_mask  = w_decision_mask  & w_in_bounds_mask  # (n_stimulus_words,)
    nw_valid_trial_mask = nw_decision_mask & nw_in_bounds_mask

    # print(w_decision_mask.shape, w_in_bounds_mask.shape)
    # print(w_letters_mask.shape, w_valid_trial_mask.shape, w_letter_layer_activities.shape)

    w_letter_activations = w_letter_layer_activities[:, w_letters_mask & w_valid_trial_mask[:, None] ]
    nw_letter_activations = nw_letter_layer_activities[:, nw_letters_mask & nw_valid_trial_mask[:, None] ]

    WS_mean_decision_index = int(np.hstack([w_decision_indices.squeeze()[w_decision_mask], nw_decision_indices.squeeze()[nw_decision_mask]]).mean())
    print(WS_mean_decision_index)
    correct_WS = w_letter_activations[WS_mean_decision_index,:].mean() > nw_letter_activations[WS_mean_decision_index,:].mean()





    # ------------------------------------------------ lexical competition --------------------------------------------------

    # for LC, we want to filter out trials where: 1. no decision was made, 2. the incorrect decision was made, and 3. activity in at least one node exited an arbitrary "in-bounds" region (e.g. [0,10])
    # --- each of these should be a boolean mask with shape (n_stimulus_words,)

    pairwise_edit_distance=analysis.get_pairwise_edit_distance(vocab, model_params.wordlist) # (n_stimulus_words, n_word_nodes)
    edit_dist0_mask = pairwise_edit_distance == 0
    edit_dist1_mask = pairwise_edit_distance == 1
    edit_dist2_mask = pairwise_edit_distance == 2
    edit_distUR_mask = pairwise_edit_distance  > 2


    w_correct_trial_mask = w_decision_mask & w_correct_decision_mask & w_in_bounds_mask

    # the 2D boolean mask along the last two dimensions of word_layer_activities results in a (n_times, n_true) shaped array
    edit_dist0_activities = w_word_layer_activities[:, edit_dist0_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)
    edit_dist1_activities = w_word_layer_activities[:, edit_dist1_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)
    edit_dist2_activities = w_word_layer_activities[:, edit_dist2_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)
    edit_distUR_activities = w_word_layer_activities[:, edit_distUR_mask * w_correct_trial_mask[:,None] ] # edit_dist_mask has shape (n_stimulus_words, n_nodes); trial_mask has shape (n_stimulus_words,)

    # if we lack one of the edit-distance neighborhoods, edit_distance trail is invalid
    if any([min(edit_dist0_activities.shape)==0, min(edit_dist1_activities.shape)==0, min(edit_dist2_activities.shape)==0, min(edit_distUR_activities.shape)==0,]):
        correct_LC = -1
    else:
        LC_mean_decision_index = int(round(w_decision_indices.squeeze()[w_decision_mask].mean()))
        correct_LC = edit_dist0_activities[LC_mean_decision_index, :].mean() > edit_dist1_activities[LC_mean_decision_index, :].mean() > edit_dist2_activities[LC_mean_decision_index, :].mean() > edit_distUR_activities[LC_mean_decision_index, :].mean()

        # plt.plot(edit_dist0_activities.mean(axis=1), c='red')
        # plt.plot(edit_dist1_activities.mean(axis=1), c='orange')
        # plt.plot(edit_dist2_activities.mean(axis=1), c='yellow')
        # plt.plot(edit_distUR_activities.mean(axis=1), c='grey')
        # plt.show()


    # ----- accuracy ----
    num_correct = int(np.sum(w_correct_trial_mask))
    valid_trials = np.sum(w_valid_trial_mask)

    accuracy = num_correct / valid_trials

    num_in_bounds = w_in_bounds_mask.sum()
    num_decisions = (w_decision_mask&w_in_bounds_mask).sum() #we're only really interested in how many decisions were made of the in-bounds trials

    return accuracy, int(num_decisions), int(correct_WS), int(correct_LC), int(num_in_bounds), maxeigval  ##acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable  #(n_stabile is the number that had activity mostly bounded, i.e. no nodes greater than x*99th_percentile(activities_over_time))

























# version used for a slightly more realistic WS setup (masking out target letters)
def SDE_timing(vocab,param_dict,times, pct=1, rng=None, filter_kwargs={"high_activity_factor":100000}): #high_activity_factor for filtering out trails where activities are unstable or get really large
    if rng is None:
        rng = np.random.default_rng()
    # rng_voc = np.random.default_rng()
    voc_size = param_dict["vocab_size"]
    vocab = rng.choice(vocab,size=voc_size,replace=False)

    model_params = glv.model_parameters.Parameters(vocab)
    sims_dir_name = "method_param_sims10000"


    model_params.set_pattern_generator(**param_dict["pattern_args"])
    if param_dict["r_args"]["wgrowth"] is None: #forcing wgrowth to be the same as lgrowth for None
        param_dict["r_args"]["wgrowth"] = param_dict["r_args"]["lgrowth"]
    r = model_params.set_r(**param_dict["r_args"])
    # print(param_dict)
    try:
        M_method = param_dict["M_args"].pop("method")
    except:
        print(param_dict["M_args"])
        raise

    if M_method =="m1":
        M = model_params.set_M_method1()
    elif M_method =="m2":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        M = model_params.set_M_method2( **param_dict["M_args"] )

    elif M_method =="m3":
        M = model_params.set_M_method3( **param_dict["M_args"] )
    elif M_method =="m4":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]
        if "ltw" not in param_dict["M_args"].keys():
            param_dict["M_args"]["ltw"] = param_dict["M_args"]["wtl"]

        M = model_params.set_M_method4( **param_dict["M_args"] )
    elif M_method =="m5":
        if param_dict["M_args"]["wdecay"] is None: #in case we want to use the same value for wdecay and ldecay, None is used to force this
            param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
        if param_dict["M_args"]["wtw"] is None:
            param_dict["M_args"]["wtw"] = param_dict["M_args"]["ltl"]

        M = model_params.set_M_method5( **param_dict["M_args"] )
    else:
        # mmethod=param_dict["M_args"]["method"]
        raise Exception(f"parameters provided for 'set_M' should be chosen from ['m1', 'm2' ,'m3', 'm4'] (not '{M_method}')")


    model_params.set_xZero_generator(**param_dict["xZero_args"])
    if param_dict["integrator_args"]["forcing_type"] != "n":
        model_params.set_forcing_generator(**param_dict["forcing_args"])

    # times = np.linspace(0,40,200)

    integrator = model_params.set_integrator(times,forcing_type=param_dict["integrator_args"]["forcing_type"])



    min_activation, max_activation = 0, 100
    # stop_thresh = "min"
    stop_thresh = 0.0001


    model_size=len(model_params.nodelist)
 
    # ----------- if we want to look at solutions to the SDE moment equations and Euler Maruyama approximation

    w_xMeans_sde, w_xVars_sde = analysis.sde_integrate(model_params, times, vocab[:1], sigma=.5, order = 4)
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=100)


    # # plt.plot(times, w_xMeans_sde[:,0,model_params.nodelist.index(vocab[0])], c='b')
    # # plt.plot(times, w_xMeans_em[:,0,model_params.nodelist.index(vocab[0])], c='r')

    # plt.plot(times, np.einsum("ijkk->ijk",w_xVars_sde)[:,0,model_params.nodelist.index(vocab[0])], c='b')

    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='r')
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=1000)
    # # indices = np.arange(0,len(times_em),len(times_em//100))
    # # w_xMeans_em, w_xVars_em, times_em = w_xMeans_em[indices], w_xVars_em[indices], times_em[indices]
    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='orange')
    # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=10000, step_size=10000)
    # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='yellow')
    # # w_xMeans_em, w_xVars_em, times_em = analysis.EM_integrate(model_params, times, vocab[:1], sigma=.5, n_sims=1000, step_size=100000)
    # # plt.plot(times_em, np.einsum("ijkk->ijk",w_xVars_em)[:,0,model_params.nodelist.index(vocab[0])], c='grey')
    # plt.show()

    return (-1,)*6




























def get_all_combos(dictionary):
    keys = list(dictionary.keys())
    vals = list(dictionary.values())
    import itertools
    all_val_combos = [list(i) for i in itertools.product(*vals)]
    # print(all_val_combos)
    all_dict_combos = [dict(zip(keys,val_combo)) for val_combo in all_val_combos]
    return all_dict_combos


def sample_ranges(dictionary, n=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    keys = list(dictionary.keys())
    vals = list(dictionary.values())
    samples = [{} for s in range(n)]
    for k,v in dictionary.items():
        if len(v)<2:
            for s in range(n):
                samples[s][k] = v[0] #same value for each dictionary
        elif len(v)==2:
            min,max = v
            for s in range(n):
                samples[s][k] = rng.uniform(low=min,high=max)
        elif len(v)>2:
            for s in range(n):
                samples[s][k] = rng.choice(v)


    return samples




def update_all_vals_in_nested_dict(input_dict, update_func, func_arg_type):
    '''recursive function that finds all terminal values in a nested dict and updates them with the user-provided "update_func"  '''
    if type(input_dict) == dict:
        for key,val in input_dict.items(): #so the initial call will loop over all keys, subsequent calls will loop over keys of nested dicts, and a terminal call will return an updated value
            input_dict[key] = update_all_vals_in_nested_dict(val,update_func,func_arg_type)
    # elif type(input_dict) in func_arg_type:
    #     return update_func(input_dict)
    else:
        return update_func(input_dict)
        # func_name = update_func.func.__name__
        # raise Exception(f"terminal dictionary values must all be of type: {func_arg_type} to be updated using '{func_name}'\nThe terminal value {input} was discovered with type: {type(input)}")
    return input_dict



# if __name__=="__main__":
def main():
    # sim_num=13 #just a marker so we can save older parameter searches, and run new ones with different parameter combos
    # sim_num=10001
    # sim_num = "06_26_2025c"
    # sim_num = "07_07_2025x"
    # sim_num = "07_08_2025" # big run with 1000 for each vocab size
    # sim_num = "07_08_2025b" # testing WS/LC plots
    # sim_num = "07_09_2025"
    sim_num = "07_10_2025" #testing postive growth rates and strong negative decay(self-interaction)


    # voc_size=100
    # vocab = np.random.choice(vocab,voc_size,replace=False)
    # n_params = 3
    # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # t = np.arange(0,20,0.01)
    t = np.linspace(0,200,1000)
    # 20, 200



    # xZero_args = {"method":["b","n"], "on":[0.1,0.7], "off":[0.01, 0.1], "sigma":[0], "seed":[1229]}
    xZero_args = {"method":["b"], "on":[0.9], "off":[0.01], "sigma":[0], "seed":[1229]}
    # xZero_args = {"method":["b"], "on":[.9], "off":[0.1], "sigma":[0], "seed":[1229]}

    # forcing_args = {"method":["gauss"], "secs_per_letter":[1/voc_size], "overlap":[0.05], "strength":[5], "delimiter":[' ']}
    # forcing_args_m = {"method":["gauss"], "secs_per_letter":[0.5], "overlap":[0.25], "strength":[10], "delimiter":[' ']}
    # integrator_args = {"forcing_type":["n","a","m"]}
    # integrator_args = {"forcing_type":["n","m"]}
    integrator_args = {"forcing_type":["n"]}

    # pattern_args = {"pattern_on":[0.1],"pattern_off":[0.01]}
    # pattern_args = {"pattern_on":[0.9],"pattern_off":[0.01]}

    pattern_args = {"pattern_on":[0.99],"pattern_off":[.01]}
    # pattern_args = {"pattern_on":[0.99],"pattern_off":[.01]}

    # r_args = {"lgrowth":[0.01, 0.1, 0.3],"wgrowth":[0.04,0.1,0.2,0.3]}#[(0.01,0.04),(0.3,0.3),(0.3,0.2),(0.3,0.1),(0.1,0.3),(0.1,0.2),(0.1,0.1)]}  #(0.3#0.1,0.3#0.2#0.1)
    # r_args = {"lgrowth":[0.01, 0.3],"wgrowth":[0.04,0.3]}
    # r_args = {"lgrowth":list(np.linspace(-1,1,n_params)),"wgrowth":list(np.linspace(-1,1,n_params))}   #testing negative growth rates
    # r_args = {"lgrowth":[-1,-0.00001],"wgrowth":[-1,-0.00001]}   #testing negative growth rates
    # r_args = {"lgrowth":[-0.1,-0.00001],"wgrowth":[-0.1,-0.00001]}   #testing negative growth rates
    # r_args = {"lgrowth":[-1,1],"wgrowth":[-1,1]} 
    r_args = {"lgrowth":[0,10],"wgrowth":[0,10]} 

    # #temporary testing ------------comment out after :
    # forcing_args = {"method":["gauss+noise"], "secs_per_letter":[1/np.log(voc_size)/np.log(10)], "overlap":[0.05], "strength":[10], "delimiter":[' ']}
    # M4_args = {"method":["m5"], "ldecay":[-1], "wdecay":[-1], "ltl":[-9], "wtl":[0.2], "ltw":[0.2], "wtw":[-5], "fn":[lambda n: np.log(n)/np.log(10)]}
    # r_args = {"lgrowth":[-1],"wgrowth":[-1]}

    # r_args = {"lgrowth":[0.01],"wgrowth":[0.04]}
    M1_args = {"method":["m1"]}
    # M2_args = {"method":["m2"], "ldecay":list(np.linspace(-1,0.1,n_params)), "wdecay":list(np.linspace(-1,0.1,n_params)), "ltl":list(np.linspace(-1,0.1,n_params)), "wtw":list(np.linspace(-1,0.1,n_params)), "fn":[lambda n: np.log(n)/np.log(10)]}
    # M3_args = {"method":["m3"], "eps":list(np.linspace(0.1,0.2,n_params)), "eps2":list(np.linspace(0.01,0.3,3))}
    # M4_args = {"method":["m4"], "ldecay":list(np.linspace(-1,0.1,n_params)), "wdecay":list(np.linspace(-1,0.1,n_params)), "ltl":list(np.linspace(-1,0.1,n_params)), "wtl":list(np.linspace(0.1,1,n_params)), "ltw":list(np.linspace(0.1,1,n_params)), "wtw":list(np.linspace(-1,0.1,n_params)), "fn":[lambda n: np.log(n)/np.log(10)]}

    # M2_args = {"method":["m2"], "ldecay":[-1,-0.1], "wdecay":[-1,-0.1], "ltl":[-1,-0.1], "wtw":[-1,-0.1], "fn":[lambda n: np.log(n)/np.log(10)]}
    # M3_args = {"method":["m3"], "eps":[0.1,0.2], "eps2":[0.01,3]}
    # M4_args = {"method":["m4"], "ldecay":[-1,-0.1], "wdecay":[-1,-0.1], "ltl":[-1,-0.1], "wtl":[0.1,1], "ltw":[0.1,1], "wtw":[-1,-0.1], "fn":[lambda n: np.log(n)/np.log(10)]}

    M2_args = {"method":["m2"], "ldecay":[-2,-1], "wdecay":[-2,-1], "ltl":[-1,-0.00001], "wtw":[-1,-0.00001], "fn":[lambda n: np.log(n)/np.log(10)]}
    M3_args = {"method":["m3"], "eps":[0.00001,1], "eps2":[0.00001,1]}
    M4_args = {"method":["m4"], "ldecay":[-10,-9], "wdecay":[-10,-9], "ltl":[-1,-0.00001], "wtl":[0.00001,.1], "wtw":[-1,-0.00001], "fn":[lambda n: np.log(n)/np.log(10)]}


    M_args = [M4_args]
    # [M1_args,M2_args,M3_args,M4_args]
    # M_args = [M1_args, M2_args,M3_args,M4_args]
    # M_args = [M2_args,M3_args,M4_args]


    seed=12515
    n_sample_points = 1000
    # vocab_sizes = [10,25,50,75,100,1000]
    vocab_sizes = [1000]
    rng = np.random.default_rng(seed) #reset each time, so we should get the same vocab subset and psuedoword list for each run


    all_param_dicts = []

    # for M_arg_dict in [i for option in [sample_ranges(M_arg, n=n_sample_points) for M_arg in M_args] for i in option]:
    #     for r_arg_dict in sample_ranges(r_args, n=n_sample_points):
    #         all_param_dicts += [{"M_args":M_arg_dict.copy(),
    #                             "pattern_args":pattern_arg_dict.copy(),
    #                             "r_args":r_arg_dict.copy(),
    #                             "xZero_args":xZero_arg_dict.copy(),
    #                             "forcing_args":forcing_arg_dict.copy(),
    #                             "integrator_args":integrator_arg_dict.copy()}]


    for vocab_size in vocab_sizes:



        #m1_r = {k:v for k,v in results.items() if v["M_method"]=="m1" and filter_func(v["param_dict"][filter_category][filter_name])}
        #m2_r = {k:v for k,v in results.items() if v["M_method"]=="m2" and filter_func(v["param_dict"][filter_category][filter_name])}
        #m3_r = {k:v for k,v in results.items() if v["M_method"]=="m3" and filter_func(v["param_dict"][filter_category][filter_name])}

        # # print([v["param_dict"]["vocab_size"] for v in results.values()])
        # # selected_param_vocab_size=100
        # # m4_r = {k:v for k,v in results.items() if v["M_method"]=="m4" and v["param_dict"]["vocab_size"]==f'{selected_param_vocab_size}'}# and filter_func(v["param_dict"][filter_category][filter_name])}
        # # m4_r = {k:v for k,v in m4_r.items() if not np.isnan(float(v["acc"])) and not float(v["acc"])==-1 }


        # varied_args = {'M_args': ['ldecay', 'wdecay', 'ltl', 'wtl', 'ltw', 'wtw'], 'r_args': ['lgrowth', 'wgrowth']}
        # labels = []
        # from collections import defaultdict
        # best_params = defaultdict(dict)
        # worst_params = defaultdict(dict)

        # for key in varied_args.keys():
        #     for arg in varied_args[key]:
        #         labels.append(arg)
        #         vals1 = np.array([float(ri["param_dict"][key][arg]) for ri in m4_r.values() if  float(ri["acc"])>=0.99 and float(ri["correct_word_sup"])==1 and float(ri["correct_edit_dist"])==1 ])
        #         #vals1 = np.array([float(ri["param_dict"][key][arg]) for ri in m4_r.values() if ops[reference_op1](float(ri[reference_arg1]),reference_val1) ])
        #         best_params[key][arg] = vals1
        #         vals2 = np.array([float(ri["param_dict"][key][arg]) for ri in m4_r.values() if  float(ri["acc"])<0.99 and (float(ri["correct_word_sup"])!=1 or float(ri["correct_edit_dist"])!=1) ])
        #         worst_params[key][arg] = vals2







        # selected_params = {key:{argval:np.median(arr) for argval,arr in val.items()} for key,val in best_params.items()}
        # # selected_params[]







        for M_arg in M_args:

            M_arg_samples = sample_ranges(M_arg.copy(), n=n_sample_points)
            pattern_arg_samples = sample_ranges(pattern_args.copy(), n=n_sample_points)
            r_arg_samples = sample_ranges(r_args.copy(), n=n_sample_points)
            xZero_arg_samples = sample_ranges(xZero_args.copy(), n=n_sample_points)
            integrator_arg_samples = sample_ranges(integrator_args.copy(), n=n_sample_points)
            for n in range(n_sample_points):
                all_param_dicts += [{"vocab_size":vocab_size,
                                    "M_args":M_arg_samples[n],
                                    "pattern_args":pattern_arg_samples[n],
                                    "r_args":r_arg_samples[n],
                                    "xZero_args":xZero_arg_samples[n],
                                    # "forcing_args":forcing_arg_dict.copy(),
                                    "integrator_args":integrator_arg_samples[n]}]



        # for M_arg_dict in [i for option in [sample_ranges(M_arg, n=n_sample_points, rng=rng) for M_arg in M_args] for i in option]:
        #     for pattern_arg_dict in get_all_combos(pattern_args):
        #         for r_arg_dict in sample_ranges(r_args, n=n_sample_points, rng=rng):
        #             for xZero_arg_dict in get_all_combos(xZero_args):
        #                 # for forcing_arg_dict in get_all_combos(forcing_args):
        #                 for integrator_arg_dict in get_all_combos(integrator_args):
        #                     all_param_dicts += [{"vocab_size":vocab_size,
        #                                         "M_args":M_arg_dict.copy(),
        #                                         "pattern_args":pattern_arg_dict.copy(),
        #                                         "r_args":r_arg_dict.copy(),
        #                                         "xZero_args":xZero_arg_dict.copy(),
        #                                         # "forcing_args":forcing_arg_dict.copy(),
        #                                         "integrator_args":integrator_arg_dict.copy()}]


    print("#########################################")
    # print("param_dict lengths: ",len(all_param_dicts))
    # exit()
    # print(all_param_dicts[:3])
    # all_param_dicts = [dict(d.copy()) for d in all_param_dicts]
    # exit()
    # i=0
    xi=[0,0,0,0]
    xv=0
    results = {}

    
    # selected_params["M_args"]["method"]="m4"
    # selected_params["M_args"]["fn"] = lambda n: np.log(n)/np.log(10)

    # # {"method":["m4"], "ldecay":[-1,-0.1], "wdecay":[-1,-0.1], "ltl":[-1,-0.1], "wtl":[0.1,1], "ltw":[0.1,1], "wtw":[-1,-0.1], "fn":[lambda n: np.log(n)/np.log(10)]}

    # param_dict = {"vocab_size":vocab_size,
    #             "M_args":selected_params["M_args"],
    #             "pattern_args":pattern_args,
    #             "r_args":selected_params["r_args"],
    #             "xZero_args":xZero_args,
    #             # "forcing_args":forcing_arg_dict.copy(),
    #             "integrator_args":integrator_args}




    # M_method = param_dict["M_args"]["method"]
    # rng = np.random.default_rng(seed) #reset each time, so we should get the same vocab subset and psuedoword list for each run
    # acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable = full_output_for_params(vocab,param_dict,t, pct=1, rng=rng, filter_kwargs={"high_activity_factor":100000}, sim_num=sim_num)
    # print(f"\n\n-------------------- {vocab_size} --------------------")
    # # print(param_dict)
    # print()
    # results = {"M_method":M_method,"param_dict":param_dict, "acc":acc, "n_decisions":n_decisions, "correct_word_sup":correct_word_sup, "correct_edit_dist":correct_edit_dist, "n_stable":n_stable }
    # print(f"acc: {acc}, n_decisions: {n_decisions}, correct_word_sup: {correct_word_sup}, correct_edit_dist: {correct_edit_dist}, n_stable: {n_stable}")
    # print("-----------------------------------------------")

    for i,param_dict in enumerate(all_param_dicts):
        # if param_dict["M_args"]["method"]=="m1" or param_dict["M_args"]["method"]=="m2" or param_dict["M_args"]["method"]=="m3":
        #     continue
        # # print(all_param_dicts[:3])
        # if not param_dict["M_args"]["method"]==f"m{xv+1}":
        #     if param_dict["M_args"]["method"][-1]==xv+1 + 1: # to catch the case where we want more iterations than some methods have options ()
        #         xv+=1
        #     else:
        #         continue
        #
        # xi[xv]+=1
        # if xi[xv]>4:
        #     xv+=1
        # i+=1
        M_method = param_dict["M_args"]["method"]
        seed1 = rng.integers(0,10000)
        rng = np.random.default_rng(seed1) #reset each time, so we should get the same vocab subset and psuedoword list for each run
        acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable, maxeigval = full_output_for_params(vocab,param_dict,t, pct=1, rng=rng, filter_kwargs={"high_activity_factor":100000})
        # acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable, maxeigval = SDE_timing(vocab,param_dict,t, pct=1, rng=rng, filter_kwargs={"high_activity_factor":100000})
        print(f"\n\n-------------------- {i} --------------------")
        # print(param_dict)
        # print()
        results[i] = {"M_method":M_method,"param_dict":param_dict, "acc":acc, "n_decisions":n_decisions, "correct_word_sup":correct_word_sup, "correct_edit_dist":correct_edit_dist, "n_stable":n_stable , "maxeigval":maxeigval}
        print(f"acc: {acc}, n_decisions: {n_decisions}, correct_word_sup: {correct_word_sup}, correct_edit_dist: {correct_edit_dist}, n_stable: {n_stable}")
        # print("-----------------------------------------------")

        if i%100==0:
            results_partial = update_all_vals_in_nested_dict(results.copy(), str, str)
            with open(f"gLVIA_param_search_results_partial_{sim_num}.pickle","wb") as f:
                pickle.dump(results_partial,f)


    results = update_all_vals_in_nested_dict(results, str, str)

    with open(f"gLVIA_param_search_results_{sim_num}.pickle","wb") as f:
        pickle.dump(results,f)
        # f.write(str(results))
# assert len(set(vocab)) == len(vocab)
# print("n_repeted_words: ", len(vocab)-len(set(vocab)))
# for voc_size in range(100,200,100):
#
#     # vocab_subset = np.random.choice(vocab,voc_size,replace=False)
#     sent_len=5
#     vocab_subset =  [" ".join(np.random.choice(vocab,sent_len,replace=False)) for _ in range(5)]
#
#     model_params = glv.model_parameters.Parameters(vocab_subset)
#
#     M = model_params.set_M_method2(  ldecay_in = ldecay_in  ,  wdecay_in = wdecay_in  ,  ltl_in = ltl_in  ,  wtw_in = wtw_in  ,  pattern_on = pattern_on/(len(vocab_subset)/10)  ,  pattern_off = pattern_off  ,  stoch = False  )
#     r = model_params.set_r(lgrowth, wgrowth)
#
#     accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
#     t = np.arange(0,2,0.01)
#     make_word_superiority_plot(model_params,model_params.wordlist, t)
#     # make_substitution_plot(model_params,vocab_subset,accumulated_thresh)
#     # accuracy,num_leftout=get_accuracy_deterministic(model_params, t, accumulated_thresh)
#     # print("accuracy: ", accuracy, " ; number leftout: ", num_leftout)











# if __name__=="__main__":
def main2():
    # sim_num=13 #just a marker so we can save older parameter searches, and run new ones with different parameter combos
    # sim_num=10001
    # sim_num = "06_26_2025c"
    # sim_num = "07_07_2025x"
    # sim_num = "07_08_2025" # big run with 1000 for each vocab size
    # sim_num = "07_08_2025b" # testing WS/LC plots
    # sim_num = "07_09_2025" # testing median of best params from 07_08_2025
    # sim_sum = "07_10_2025" #testing postive growth rates and strong negative decay(self-interaction)
    sim_num = "07_14_2025"

    # voc_size=100
    # vocab = np.random.choice(vocab,voc_size,replace=False)
    # n_params = 3
    # accumulated_thresh=0.000001/np.log(voc_size)#/np.log(80)
    # t = np.arange(0,20,0.01)
    t = np.linspace(0,10,100)
    # 20, 200



    # xZero_args = {"method":["b","n"], "on":[0.1,0.7], "off":[0.01, 0.1], "sigma":[0], "seed":[1229]}
    xZero_args = {"method":"b", "on":0.9, "off":0.01, "sigma":0, "seed":1229}
    # xZero_args = {"method":["b"], "on":[.9], "off":[0.1], "sigma":[0], "seed":[1229]}

    # forcing_args = {"method":["gauss"], "secs_per_letter":[1/voc_size], "overlap":[0.05], "strength":[5], "delimiter":[' ']}
    # forcing_args_m = {"method":["gauss"], "secs_per_letter":[0.5], "overlap":[0.25], "strength":[10], "delimiter":[' ']}
    # integrator_args = {"forcing_type":["n","a","m"]}
    # integrator_args = {"forcing_type":["n","m"]}
    integrator_args = {"forcing_type":"n"}

    # pattern_args = {"pattern_on":[0.1],"pattern_off":[0.01]}
    # pattern_args = {"pattern_on":[0.9],"pattern_off":[0.01]}

    pattern_args = {"pattern_on":0.99,"pattern_off":.01}
    # pattern_args = {"pattern_on":[0.99],"pattern_off":[.01]}



    seed=12515
    n_sample_points = 1000
    # vocab_sizes = [10,25,50,75,100,1000]
    vocab_sizes = [50,100,1000]
    m_methods = ["m2","m3","m4"]
    # vocab_sizes = [100]
    rng = np.random.default_rng(seed) #reset each time, so we should get the same vocab subset and psuedoword list for each run


    all_param_dicts = []

    for m_num in m_methods:
        for vocab_size in vocab_sizes:

            saved_num = "07_08_2025"
            with open(rf"C:\Users\mitch\Documents\OSU\Research data and code\Lotka_Volterra project_new\gLVIA_param_search_results_partial_{saved_num}.pickle","rb") as f:
                results = pickle.load(f)

            #m1_r = {k:v for k,v in results.items() if v["M_method"]=="m1" and filter_func(v["param_dict"][filter_category][filter_name])}
            #m2_r = {k:v for k,v in results.items() if v["M_method"]=="m2" and filter_func(v["param_dict"][filter_category][filter_name])}
            #m3_r = {k:v for k,v in results.items() if v["M_method"]=="m3" and filter_func(v["param_dict"][filter_category][filter_name])}

            # print([v["param_dict"]["vocab_size"] for v in results.values()])
            selected_param_vocab_size=100
            # selected_param_vocab_size=vocab_size
            # m_num = "m4"
            mx_r = {k:v for k,v in results.items() if v["M_method"]==m_num and v["param_dict"]["vocab_size"]==f'{selected_param_vocab_size}'}# and filter_func(v["param_dict"][filter_category][filter_name])}
            mx_r = {k:v for k,v in mx_r.items() if not np.isnan(float(v["acc"])) and not float(v["acc"])==-1 }


            varied_args_all = {"m1":{'M_args': ['ldecay', 'wdecay', 'ltl', 'wtl', 'ltw', 'wtw'], 'r_args': ['lgrowth', 'wgrowth']}, 
                               "m2":{'M_args': ['ldecay', 'wdecay', 'ltl', 'wtw'], 'r_args': ['lgrowth', 'wgrowth']},
                               "m3":{'M_args': [ 'eps', 'eps2'], 'r_args': ['lgrowth', 'wgrowth']},
                               "m4":{'M_args': ['ldecay', 'wdecay', 'ltl', 'wtl', 'ltw', 'wtw'], 'r_args': ['lgrowth', 'wgrowth']}
                               }
            varied_args = varied_args_all[m_num]
            labels = []
            from collections import defaultdict
            best_params = defaultdict(dict)
            worst_params = defaultdict(dict)

            voc_size = 100
            for key in varied_args.keys():
                for arg in varied_args[key]:
                    labels.append(arg)
                    vals1 = np.array([float(ri["param_dict"][key][arg]) for ri in mx_r.values() 
                                        if  float(ri["acc"])>=0.95 
                                        and float(ri["correct_word_sup"])==1 
                                        and float(ri["correct_edit_dist"])==1    
                                        and float(ri["n_stable"])>=voc_size 
                                        # and float(ri["n_decisions"])>=.95*voc_size 
                                        and not np.isnan(float(ri["acc"])) 
                                        and float(ri["acc"])!=-1 ])
                    #vals1 = np.array([float(ri["param_dict"][key][arg]) for ri in m4_r.values() if ops[reference_op1](float(ri[reference_arg1]),reference_val1) ])
                    best_params[key][arg] = vals1
                    vals2 = np.array([float(ri["param_dict"][key][arg]) for ri in mx_r.values() 
                                        if  float(ri["acc"])<0.95 
                                        or float(ri["correct_word_sup"])!=1 
                                        or float(ri["correct_edit_dist"])!=1       
                                        or float(ri["n_stable"])<voc_size 
                                        # or float(ri["n_decisions"])<.95*voc_size 
                                        or np.isnan(float(ri["acc"])) 
                                        or float(ri["acc"])==-1   ])
                    worst_params[key][arg] = vals2







            selected_params = {key:{argval:np.median(arr) for argval,arr in val.items()} for key,val in best_params.items()}


            
            selected_params["M_args"]["method"]=m_num
            if not m_num=="m3":
                selected_params["M_args"]["fn"] = lambda n: np.log(n)/np.log(10)

            # {"method":["m4"], "ldecay":[-1,-0.1], "wdecay":[-1,-0.1], "ltl":[-1,-0.1], "wtl":[0.1,1], "ltw":[0.1,1], "wtw":[-1,-0.1], "fn":[lambda n: np.log(n)/np.log(10)]}

            param_dict = {"vocab_size":vocab_size,
                        "M_args":selected_params["M_args"],
                        "pattern_args":pattern_args,
                        "r_args":selected_params["r_args"],
                        "xZero_args":xZero_args,
                        # "forcing_args":forcing_arg_dict.copy(),
                        "integrator_args":integrator_args}

            # if "wdecay" not in param_dict["M_args"].keys():
            #     param_dict["M_args"]["wdecay"] = param_dict["M_args"]["ldecay"]
            # if m_num == "m4" and "ltw" not in param_dict["M_args"]:
            #     param_dict["M_args"]["ltw"] = param_dict["M_args"]["wtl"]
            print(param_dict)


            M_method = param_dict["M_args"]["method"]
            rng = np.random.default_rng(seed) #reset each time, so we should get the same vocab subset and psuedoword list for each run
            # acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable = full_output_for_params(vocab,param_dict,t, pct=1, rng=rng, filter_kwargs={"high_activity_factor":100000})
            acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable, maxeigval = full_output_for_params(vocab,param_dict,t, pct=1, rng=rng, filter_kwargs={"high_activity_factor":100000})
            # acc, n_decisions, correct_word_sup, correct_edit_dist, n_stable, maxeigval = SDE_timing(vocab,param_dict,t, pct=1, rng=rng, filter_kwargs={"high_activity_factor":100000})
            print(f"\n\n-------------------- {vocab_size}, {m_num} --------------------")
            # print(param_dict)
            # print()
            # results[i] = {"M_method":M_method,"param_dict":param_dict, "acc":acc, "n_decisions":n_decisions, "correct_word_sup":correct_word_sup, "correct_edit_dist":correct_edit_dist, "n_stable":n_stable , "maxeigval":maxeigval}
            print(f"acc: {acc}, n_decisions: {n_decisions}, correct_word_sup: {correct_word_sup}, correct_edit_dist: {correct_edit_dist}, n_stable: {n_stable}")
            # print("-----------------------------------------------")

            # print(f"\n\n-------------------- {vocab_size} --------------------")
            # # print(param_dict)
            # print()
            # results = {"M_method":M_method,"param_dict":param_dict, "acc":acc, "n_decisions":n_decisions, "correct_word_sup":correct_word_sup, "correct_edit_dist":correct_edit_dist, "n_stable":n_stable }
            # print(f"acc: {acc}, n_decisions: {n_decisions}, correct_word_sup: {correct_word_sup}, correct_edit_dist: {correct_edit_dist}, n_stable: {n_stable}")
            # print("-----------------------------------------------")











from line_profiler import LineProfiler


if __name__=="__main__":


    # profiler = LineProfiler()
    # # profiler.add_function(glv.SDEs.integrate_sde.Integrator.integrate_CMN)  # Add the method to the profiler
    # profiler.add_function(glv.SDEs.delta_moments_sde.ddt_moments_CMN.ddt_moments_from_arrays)
    # profiler.enable()

    main2()

    # profiler.print_stats()