"""
Loads Switchboard dataset.
Function is taken from https://github.com/ilimugur/short-text-classification
"""

from swda.swda import CorpusReader


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

    print('Loaded SwDA Corpus.')
    return talks, talk_names, tag_indices, tag_occurances


if __name__ == '__main__':
    talks, talk_names, tag_indices, tag_occurances = load_swda_corpus_data('../swda/swda/swda')
    import ipdb; ipdb.set_trace()
