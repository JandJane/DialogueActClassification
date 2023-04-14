"""
Loads Switchboard dataset.
Function is taken from https://github.com/ilimugur/short-text-classification
"""

from swda.swda import CorpusReader

TAG_MAP = {
    'sd': 'Statement-non-opinion',
    'b': 'Acknowledge (Backchannel)',
    'sv': 'Statement-opinion',
    'aa': 'Agree/Accept',
    '%': 'Uninterpretable',
    'ba': 'Appreciation',
    'qy': 'Yes-No-Question',
    'x': 'Non-verbal',
    'ny': 'Yes answers',
    'fc': 'Conventional-closing',
    'qw': 'Wh-Question',
    'nn': 'No answers',
    'bk': 'Response Acknowledgement',
    'h': 'Hedge',
    'qy^d': 'Declarative Yes-No-Question',
    'bh': 'Backchannel in question form',
    '^q': 'Quotation',
    'bf': 'Summarize/reformulate',
    'na': 'Affirmative non-yes answers',
    'ad': 'Action-directive',
    '^2': 'Collaborative Completion',
    'b^m': 'Repeat-phrase',
    'qo': 'Open-Question',
    'qh': 'Rhetorical-Questions',
    '^h': 'Hold before answer/agreement',
    'ar': 'Reject',
    'ng': 'Negative non-no answers',
    'br': 'Signal-non-understanding',
    'no': 'Other answers',
    'fp': 'Conventional-opening',
    'qrr': 'Or-Clause',
    'arp_nd': 'Dispreferred answers',
    't3': '3rd-party-talk',
    'oo_co_cc': 'Offers, Options, Commits',
    't1': 'Self-talk',
    'bd': 'Downplayer',
    'aap_am': 'Maybe/Accept-part',
    '^g': 'Tag-Question',
    'qw^d': 'Declarative Wh-Question',
    'fa': 'Apology',
    'ft': 'Thanking',
    '+': '+',
    'fo_o_fw_"_by_bc': 'Other',
}


def load_swda_corpus_data(swda_directory):
    print('Loading SwDA Corpus...')
    corpus_reader = CorpusReader(swda_directory)

    talks = []
    talk_names = []
    tags_seen = set()
    tag_occurances = {}
    for transcript in corpus_reader.iter_transcripts(False):
        name = 'sw' + str(transcript.conversation_no)
        talk_names.append(name)
        conversation_content = []
        conversation_tags = []
        for utterance in transcript.utterances:
            conversation_content.append(utterance.text_words(True))
            tag = utterance.damsl_act_tag()
            conversation_tags.append(tag)
            if tag not in tags_seen:
                tags_seen.add(tag)
                tag_occurances[tag] = 1
            else:
                tag_occurances[tag] += 1
        talks.append((conversation_content, conversation_tags))

    print('\nFound ' + str(len(tags_seen)) + ' different utterance tags.\n')

    tag_indices = {tag: i for i, tag in enumerate(sorted(list(tags_seen)))}

    for talk in talks:
        talk_tags = talk[1]
        for i, tag in enumerate(talk_tags):
            talk_tags[i] = tag_indices[tag]
            
    tag_indices = {TAG_MAP[k]: v for k, v in tag_indices.items()}
    tag_occurances = {TAG_MAP[k]: v for k, v in tag_occurances.items()}

    print('Loaded SwDA Corpus.')
    return talks, talk_names, tag_indices, tag_occurances


if __name__ == '__main__':
    talks, talk_names, tag_indices, tag_occurances = load_swda_corpus_data('../swda/swda/swda')
    import ipdb; ipdb.set_trace()
