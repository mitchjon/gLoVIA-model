import numpy as np
import os
import pickle


import gLoVIA_parametrization as glv
import analysis_utils as analysis




vocab = ['a', 'able', 'about', 'above', 'across', 'act', 'actor', 'active', 'activity', 'add', 'afraid', 'after', 'again', 'age', 'ago', 'agree', 'air', 'all', 'alone', 'along', 'already', 'always', 'am', 'amount', 'an', 'and', 'angry', 'another', 'answer', 'any', 'anyone', 'anything', 'anytime', 'appear', 'apple', 'are', 'area', 'arm', 'army', 'around', 'arrive', 'art', 'as', 'ask', 'at', 'attack', 'aunt', 'autumn', 'away', 'baby', 'base', 'back', 'bad', 'bag', 'ball', 'bank', 'basket', 'bath', 'be', 'bean', 'bear', 'beautiful', 'beer', 'bed', 'bedroom', 'behave', 'before', 'begin', 'behind', 'bell', 'below', 'besides', 'best', 'better', 'between', 'big', 'bird', 'birth', 'birthday', 'bit', 'bite', 'black', 'bleed', 'block', 'blood', 'blow', 'blue', 'board', 'boat', 'body', 'boil', 'bone', 'book', 'border', 'born', 'borrow', 'both', 'bottle', 'bottom', 'bowl', 'box', 'boy', 'branch', 'brave', 'bread', 'break', 'breakfast', 'breathe', 'bridge', 'bright', 'bring', 'brother', 'brown', 'brush', 'build', 'burn', 'business', 'bus', 'busy', 'but', 'buy', 'by', 'bundle', 'cake', 'call', 'can', 'candle', 'cap', 'car', 'card', 'care', 'careful', 'careless', 'carry', 'case', 'cat', 'catch', 'central', 'century', 'certain', 'chair', 'chance', 'change', 'chase', 'cheap', 'cheese', 'chicken', 'child', 'children', 'chocolate', 'choice', 'choose', 'circle', 'city', 'class', 'clever', 'clean', 'clear', 'climb', 'clock', 'cloth', 'clothes', 'cloud', 'cloudy', 'close', 'coffee', 'coat', 'coin', 'cold', 'collect', 'colour', 'comb', 'come' , 'comfortable', 'common', 'compare', 'complete', 'computer', 'condition', 'continue', 'control', 'cook', 'cool', 'copper', 'corn', 'corner', 'correct', 'cost', 'contain', 'count', 'country', 'course', 'cover', 'crash', 'cross', 'cry', 'cup', 'cupboard', 'cut', 'dance', 'danger', 'dangerous', 'dark', 'daughter', 'day', 'dead', 'decide', 'decrease', 'deep', 'deer', 'depend', 'desk', 'destroy', 'develop', 'die', 'different', 'difficult', 'dinner', 'direction', 'dirty', 'discover', 'dish', 'do', 'dog', 'door', 'double', 'down', 'draw', 'dream', 'dress', 'drink', 'drive', 'drop', 'dry', 'duck', 'dust', 'duty', 'destroy', 'dedicated', 'each', 'ear', 'early', 'earn', 'earth', 'east', 'easy', 'eat', 'education', 'effect', 'egg', 'eight', 'either', 'electric', 'elephant', 'else', 'empty', 'end', 'enemy', 'enjoy', 'enough', 'enter', 'equal', 'entrance', 'escape', 'even', 'evening', 'event', 'ever', 'every', 'everyone', 'exact', 'everybody', 'examination', 'example', 'except', 'excited', 'exercise', 'expect', 'expensive', 'explain', 'extremely', 'eye', 'face', 'fact', 'fail', 'fall', 'false', 'family', 'famous', 'far', 'farm', 'father', 'fast', 'fat', 'fault', 'fear', 'feed', 'feel', 'female', 'fever', 'few', 'fight', 'fill', 'film', 'find', 'fine', 'finger', 'finish', 'fire', 'first', 'fit', 'five', 'fix', 'flag', 'flat', 'float', 'floor', 'flour', 'flower', 'fly', 'fold', 'food', 'fool', 'foot', 'football', 'for', 'force', 'foreign', 'forest', 'forget', 'forgive', 'fork', 'form', 'fox', 'four', 'free', 'freedom', 'freeze', 'fresh', 'friend', 'friendly', 'from', 'front', 'fruit', 'full', 'fun', 'funny', 'furniture', 'further', 'future', 'game', 'garden', 'gate', 'general', 'gentleman', 'get', 'gift', 'give', 'glad', 'glass', 'go', 'goat', 'god', 'gold', 'good', 'goodbye', 'grandfather', 'grandmother', 'grass', 'grave', 'great', 'green', 'grey', 'ground', 'group', 'grow', 'gun', 'hair', 'half', 'hall', 'hammer', 'hand', 'happen', 'happy', 'hard', 'hat', 'hate', 'have', 'he', 'head', 'healthy', 'hear', 'heavy', 'hello', 'help', 'heart', 'heaven', 'height', 'hen', 'her', 'here', 'hers', 'hide', 'high', 'hill', 'him', 'his', 'hit', 'hobby', 'hold', 'hole', 'holiday', 'home', 'hope', 'horse', 'hospital', 'hot', 'hotel', 'house', 'how', 'hundred', 'hungry', 'hour', 'hurry', 'husband', 'hurt', 'I', 'ice', 'idea', 'if', 'important', 'in', 'increase', 'inside', 'into', 'introduce', 'invent', 'iron', 'invite', 'is', 'island', 'it', 'its', 'job', 'join', 'juice', 'jump', 'just', 'keep', 'key', 'kid', 'kill', 'kind', 'king', 'kitchen', 'knee', 'knife', 'knock', 'know', 'ladder', 'lady', 'lamp', 'land', 'large', 'last', 'late', 'lately', 'laugh', 'lazy', 'lead', 'leaf', 'learn', 'leave', 'leg', 'left', 'lend', 'length', 'less', 'lesson', 'let', 'letter', 'library', 'lie', 'life', 'light', 'like', 'lion', 'lip', 'list', 'listen', 'little', 'live', 'lock', 'lonely', 'long', 'look', 'lose', 'lot', 'love', 'low', 'lower', 'luck', 'machine', 'main', 'make', 'male', 'man', 'many', 'map', 'mark', 'market', 'marry', 'matter', 'may', 'me', 'meal', 'mean', 'measure', 'meat', 'medicine', 'meet', 'member', 'mention', 'method', 'middle', 'milk', 'mill', 'million', 'mind', 'mine', 'minute', 'miss', 'mistake', 'mix', 'model', 'modern', 'moment', 'money', 'monkey', 'month', 'moon', 'more', 'morning', 'most', 'mother', 'mountain', 'mouse', 'mouth', 'move', 'much', 'music', 'must', 'my', 'name', 'narrow', 'nation', 'nature', 'near', 'nearly', 'neck', 'need', 'needle', 'neighbour', 'neither', 'net', 'never', 'new', 'news', 'newspaper', 'next', 'nice', 'night', 'nine', 'no', 'noble', 'noise', 'none', 'nor', 'north', 'nose', 'not', 'nothing', 'notice', 'now', 'number', 'obey', 'object', 'ocean', 'of', 'off', 'offer', 'office', 'often', 'oil', 'old', 'on', 'one', 'only', 'open', 'opposite', 'or', 'orange', 'order', 'other', 'our', 'out', 'outside', 'over', 'own', 'page', 'pain', 'paint', 'pair', 'pan', 'paper', 'parent', 'park', 'part', 'partner', 'party', 'pass', 'past', 'path', 'pay', 'peace', 'pen', 'pencil', 'people', 'pepper', 'per', 'perfect', 'period', 'person', 'petrol', 'photograph', 'piano', 'pick', 'picture', 'piece', 'pig', 'pill', 'pin', 'pink', 'place', 'plane', 'plant', 'plastic', 'plate', 'play', 'please', 'pleased', 'plenty', 'pocket', 'point', 'poison', 'police', 'polite', 'pool', 'poor', 'popular', 'position', 'possible', 'potato', 'pour', 'power', 'present', 'press', 'pretty', 'prevent', 'price', 'prince', 'prison', 'private', 'prize', 'probably', 'problem', 'produce', 'promise', 'proper', 'protect', 'provide', 'public', 'pull', 'punish', 'pupil', 'push', 'put', 'queen', 'question', 'quick', 'quiet', 'quite', 'radio', 'rain', 'rainy', 'raise', 'reach', 'read', 'ready', 'real', 'really', 'receive', 'record', 'red', 'remember', 'remind', 'remove', 'rent', 'repair', 'repeat', 'reply', 'report', 'rest', 'restaurant', 'result', 'return', 'rice', 'rich', 'ride', 'right', 'ring', 'rise', 'road', 'rob', 'rock', 'room', 'round', 'rubber', 'rude', 'rule', 'ruler', 'run', 'rush', 'sad', 'safe', 'sail', 'salt', 'same', 'sand', 'save', 'say', 'school', 'science', 'scissors', 'search', 'seat', 'second', 'see', 'seem', 'sell', 'send', 'sentence', 'serve', 'seven', 'several', 'sex', 'shade', 'shadow', 'shake', 'shape', 'share', 'sharp', 'she', 'sheep', 'sheet', 'shelf', 'shine', 'ship', 'shirt', 'shoe', 'shoot', 'shop', 'short', 'should', 'shoulder', 'shout', 'show', 'sick', 'side', 'signal', 'silence', 'silly', 'silver', 'similar', 'simple', 'single', 'since', 'sing', 'sink', 'sister', 'sit', 'six', 'size', 'skill', 'skin', 'skirt', 'sky', 'sleep', 'slip', 'slow', 'small', 'smell', 'smile', 'smoke', 'snow', 'so', 'soap', 'sock', 'soft', 'some', 'someone', 'something', 'sometimes', 'son', 'soon', 'sorry', 'sound', 'soup', 'south', 'space', 'speak', 'special', 'speed', 'spell', 'spend', 'spoon', 'sport', 'spread', 'spring', 'square', 'stamp', 'stand', 'star', 'start', 'station', 'stay', 'steal', 'steam', 'step', 'still', 'stomach', 'stone', 'stop', 'store', 'storm', 'story', 'strange', 'street', 'strong', 'structure', 'student', 'study', 'stupid', 'subject', 'substance', 'successful', 'such', 'sudden', 'sugar', 'suitable', 'summer', 'sun', 'sunny', 'support', 'sure', 'surprise', 'sweet', 'swim', 'sword', 'table', 'take', 'talk', 'tall', 'taste', 'taxi', 'tea', 'teach', 'team', 'tear', 'telephone', 'television', 'tell', 'ten', 'tennis', 'terrible', 'test', 'than', 'that', 'the', 'their', 'theirs', 'then', 'there', 'therefore', 'these', 'thick', 'thin', 'thing', 'think', 'third', 'this', 'those', 'though', 'threat', 'three', 'tidy', 'tie', 'title', 'to', 'today', 'toe', 'together', 'tomorrow', 'tonight', 'too', 'tool', 'tooth', 'top', 'total', 'touch', 'town', 'train', 'tram', 'travel', 'tree', 'trouble', 'true', 'trust', 'twice', 'try', 'turn', 'type', 'uncle', 'under', 'understand', 'unit', 'until', 'up', 'use', 'useful', 'usual', 'usually', 'vegetable', 'very', 'village', 'voice', 'visit', 'value', 'vacuum', 'vampire', 'verb', 'validation', 'wait', 'wake', 'walk', 'want', 'warm', 'wash', 'waste', 'watch', 'water', 'way', 'we', 'weak', 'wear', 'weather', 'wedding', 'week', 'weight', 'welcome', 'well', 'west', 'wet', 'what', 'wheel', 'when', 'where', 'which', 'while', 'white', 'who', 'why', 'wide', 'wife', 'wild', 'will', 'win', 'wind', 'window', 'wine', 'winter', 'wire', 'wise', 'wish', 'with', 'without', 'woman', 'wonder', 'word', 'work', 'world', 'worry', 'worst', 'write', 'wrong', 'year', 'yellow', 'yes', 'yesterday', 'yet', 'you', 'young', 'your', 'yours', 'zero', 'zoo', 'zoom']
vocab=list(set(vocab))





def full_output_for_params(vocab, voc_size, param_dict, times, M_method, rng=None, plot_LC=False, plot_WS=False): 


    # setting up a model with the specified parameters
    if rng is None:
        rng = np.random.default_rng()
    vocab = rng.choice(vocab,size=voc_size,replace=False)

    model_params = glv.gLoVIA_Parameters(vocab, node_sort = 'permutation')
    model_params.set_pattern_generator(pattern_on=param_dict['pattern_on'], pattern_off=param_dict['pattern_off'])
    model_params.set_r(lgrowth=param_dict['lgrowth'], wgrowth=param_dict['wgrowth'])

    if M_method =="m1":
        M = model_params.set_M_method1()
    elif M_method =="m2":
        M = model_params.set_M_method2( ldecay=param_dict['ldecay'],wdecay=param_dict['wdecay'],ltl=param_dict['ltl'],wtw=param_dict['wtw'] )
    elif M_method =="m3":
        M = model_params.set_M_method3( eps=param_dict['eps'],eps2=param_dict['eps2'])
    elif M_method =="m4":
        M = model_params.set_M_method4( ldecay=param_dict['ldecay'],wdecay=param_dict['wdecay'], ltl=param_dict['ltl'], wtw=param_dict['wtw'], wtl=param_dict['wtl'], ltw=param_dict['ltw'] )
    else:
        raise Exception(f"Valid M_methods are: ['m1', 'm2' ,'m3', 'm4']. Got '{M_method}'.")

    model_params.set_xZero_generator(xZero_on=param_dict['xZero_on'], xZero_off=param_dict['xZero_off'])



    # ----- compute eigenvalues and report the max 
    eigenvalues, eigenvectors = np.linalg.eig((M + M.T)/2)
    maxeigval = eigenvalues.real.max()


    # the following values are hardcoded, but we didn't extensively tune them; 
    # min/max_activation indicate the "in_bounds" activation values - trials with 
    # activity not in bounds are excluded from WS/LC calculations and the number
    # in bounds is reported in some of our plots/tables
    min_activation, max_activation = 0, 100
    stop_threshs = {50:0.002, 100:0.001, 1000:0.00001} # theshold determining when a "decision" is made by the model - when the cumulative gap between the highest activation and next highest exceeds the threshold
    stop_thresh = stop_threshs[voc_size]

    model_size=len(model_params.nodelist)



    # we'll use the xHistory computed for the full vocab for both WS and LC

    # ----- integrate for word-stimuli
    # w_xHistories_ia = model_params.batch_integrate_IA(times, vocab).transpose(0,2,1)
    # w_letter_layer_activities_ia = w_xHistories_ia[:,:, :len(model_params.letterlist)]
    # w_word_layer_activities_ia = w_xHistories_ia[:,:, -len(model_params.wordlist):]

    w_xHistories = model_params.batch_integrate(times, vocab).transpose(0,2,1)
    w_letter_layer_activities = w_xHistories[:,:, :len(model_params.letterlist)]
    w_word_layer_activities = w_xHistories[:,:, -len(model_params.wordlist):]


    w_decision_indices = analysis.compute_cumulative_gap_decision_indices(w_word_layer_activities, times, stop_thresh=stop_thresh, time_axis = 0, node_axis=-1, keepdims=True) #(1,n_stimulis_words,1)

    w_decision_times = np.take_along_axis(times[ : , None, None ] , w_decision_indices, axis=0) # times: (n_times,1,1), word_decision_indices: (1,n_stimulis_words,1) -> ()
    w_decision_activations = np.take_along_axis(w_xHistories, w_decision_indices, axis=0).squeeze(axis=0) # node activations for each batch at the decision time for that batch; xHistory: (n_times, n_stimulus_words, n_nodes), word_decision_indices: (1,n_stimulis_words,1) ->(1, n_stimulus_words, n_nodes)->(n_stimulus_words, n_nodes)
    w_predicted_nodes = w_decision_activations[:,-len(model_params.wordlist):].argmax(axis=-1) # the node (for each stimulus word) that had the highest activation at the dicision time for that stimulus word

    w_decision_mask = w_decision_indices.squeeze() != -1 # -1 is used to indicate where the stop_thresh was never exceded (we may exclude these trials, or use the final index as the decision time - in which case we can just use decision_indices as is since -1 will index to the last element)
    w_correct_decision_mask = [w_predicted_nodes[i]==model_params.wordlist.index(vocab[i]) for i in range(len(vocab))]

    w_in_bounds_mask = np.all( (w_xHistories >= min_activation) & (w_xHistories <= max_activation),  axis=(0,-1) ) #i.e. True if every node across all times is <=max and >=min for a given stimulus word





    # ----- integrate for nonword-stimuli
    nonword_list = analysis.make_nonword_list(model_params.letterlist,vocab,n_pseudowords=None, rng=None) # matches the empirical distribution of lengths, and pulls letters from those that the model already has nodes for

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


    if (np.any(w_decision_mask & w_in_bounds_mask)) and (np.any(nw_decision_mask & nw_in_bounds_mask)):
        # ----------------------------------------------- word superiority -----------------------------------------------

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

        w_letter_activations = w_letter_layer_activities[:, w_letters_mask & w_valid_trial_mask[:, None] ]
        nw_letter_activations = nw_letter_layer_activities[:, nw_letters_mask & nw_valid_trial_mask[:, None] ]

        WS_mean_decision_index = int(np.hstack([w_decision_indices.squeeze()[w_decision_mask], nw_decision_indices.squeeze()[nw_decision_mask]]).mean())
        correct_WS = w_letter_activations[WS_mean_decision_index,:].mean() > nw_letter_activations[WS_mean_decision_index,:].mean()


        if plot_WS:
            import scipy.stats as sps
            import matplotlib.pyplot as plt
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
            plt.savefig(os.path.join("plots", f"WS_{M_method}_vocsize-{voc_size}_thresh-{stop_thresh}.png"), dpi=1000, transparent=True)
            plt.clf()



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




        # ------------- plot LC ---------------
        if plot_LC:
            import scipy.stats as sps
            import matplotlib.pyplot as plt
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
                ax.plot(times, data.mean(axis=1), color = color, linestyle=linestyle, label=label)

            ax.legend(loc="upper right")
            plt.savefig(os.path.join("plots", f"LC_{M_method}_vocsize-{voc_size}_thresh-{stop_thresh}.png"), dpi=1000, transparent=True)
            plt.clf()

      # ----- accuracy ----
        num_correct = int(np.sum(w_correct_trial_mask))
        valid_trials = np.sum(w_valid_trial_mask)

        accuracy = num_correct / valid_trials

    else:
        # invalid WS/LC trials since either no decisions were made by the model, or all stimulus-words resulted in out-of-bounds activities
        correct_WS = -1
        correct_LC = -1

        accuracy = -1




  

    num_in_bounds = w_in_bounds_mask.sum()
    num_decisions = (w_decision_mask&w_in_bounds_mask).sum() #we're only really interested in how many decisions were made of the in-bounds trials


    results = {"M_method":M_method,
               "param_dict":param_dict, 
               "acc":accuracy, 
               "num_decisions":int(num_decisions), 
               "correct_WS":int(correct_WS), 
               "correct_LC":int(correct_LC), 
               "num_in_bounds":int(num_in_bounds), 
               "maxeigval":maxeigval}

    return results
















def sample_ranges(dictionary, n=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    samples = [{} for s in range(n)]
    for s in range(n):
        for k,v in dictionary.items():
            if len(v)==1:
                low=v[0]
                high=low
            else:
                low,high = v
            samples[s][k] = rng.uniform( low=low,high=high)

    return samples




if __name__=="__main__":
# def main():

    sim_num = "07_10_2025" 


    t = np.linspace(0,200,1000)


    M1_args = {"lgrowth":[0,10],
               "wgrowth":[0,10], 
               "pattern_on":[0.99],
               "pattern_off":[.01], 
               "xZero_on":[0.9], 
               "xZero_off":[0.01] }

    M2_args = {"ldecay":[-2,-1], 
               "wdecay":[-2,-1], 
               "ltl":[-1,-0.00001], 
               "wtw":[-1,-0.00001],
               "lgrowth":[0,10],
               "wgrowth":[0,10], 
               "pattern_on":[0.99],
               "pattern_off":[.01], 
               "xZero_on":[0.9],
               "xZero_off":[0.01] }

    M3_args = {"eps":[0.00001,1], 
               "eps2":[0.00001,1],
               "lgrowth":[0,10],
               "wgrowth":[0,10], 
               "pattern_on":[0.99],
               "pattern_off":[.01], 
               "xZero_on":[0.9], 
               "xZero_off":[0.01] }

    M4_args = {"ldecay":[-10,-9], 
               "wdecay":[-10,-9], 
               "ltl":[-1,-0.00001], 
               "wtl":[0.00001,.1], 
               "wtw":[-1,-0.00001],
               "lgrowth":[0,10],
               "wgrowth":[0,10], 
               "pattern_on":[0.99],
               "pattern_off":[.01], 
               "xZero_on":[0.9], 
               "xZero_off":[0.01] }


    rng = np.random.default_rng(seed=15318)

    n_sample_points = 10

    M_args_list = [('m1',M1_args), ('m2',M2_args), ('m3',M3_args), ('m4',M4_args)]
    vocab_sizes = [50,100,1000]
    for voc_size in vocab_sizes:
        for M_name, M_args in M_args_list:

            all_results = {}
            best_params = {}
            worst_params = {}

            param_dicts = sample_ranges(M_args, n=n_sample_points)

            for i,param_dict in enumerate(param_dicts):
                results = full_output_for_params(vocab, voc_size, param_dict,t, M_name, rng=rng, plot_LC=False, plot_WS=False)
                print(f"\n\n-------------------- {i} --------------------")
                all_results[i] = results


                if results["acc"]>=95 and not np.isnan(results["acc"]) and not results["acc"]==-1\
                    and results["correct_WS"]==1\
                    and results["correct_edit_dist"]==1\
                    and results["num_in_bounds"]==len(vocab):
                    best_params[i] = results
                else:
                    worst_params[i] = results


            with open(os.path.join("data",f"gLoVIA_param_search_results_{sim_num}_vocsize-{voc_size}.pickle"),"wb") as f:
                pickle.dump(all_results,f)
            with open(os.path.join("data",f"gLoVIA_best_params_{sim_num}_{M_name}_vocsize-{voc_size}.pickle"),"wb") as f:
                pickle.dump(best_params,f)
            with open(os.path.join("data",f"gLoVIA_worst_params_{sim_num}_{M_name}_vocsize-{voc_size}.pickle"),"wb") as f:
                pickle.dump(worst_params,f)


            median_of_best_params = {k:np.median([ri[k] for ri in best_params.values()]) for k in M_args.keys()}

            with open(os.path.join("data",f"gLoVIA_median_of_best_params_{sim_num}_{M_name}_vocsize-{voc_size}.pickle"),"wb") as f:
                pickle.dump(median_of_best_params,f)


        # now plot WS/LC results for a run using median of best params
        results = full_output_for_params(vocab, median_of_best_params , t, 'm1', rng=rng, plot_LC=True, plot_WS=True)

        with open(os.path.join("data",f"gLoVIA_median_of_best_params-results-m1_vocsize-{voc_size}.pickle"),"wb") as f:
            pickle.dump(results,f)
