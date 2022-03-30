from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import string
import re

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:

    # Begin your code (Part 0)
    # Turn to lowercase
    lowercase_text = text.lower()
    # lemmatizer = WordNetLemmatizer()
    tokenizer = ToktokTokenizer()

    # remove stopwords
    # preprocessed_text = remove_stopwords(lowercase_text)

    # tokens = tokenizer.tokenize(preprocessed_text)
    tokens = tokenizer.tokenize(lowercase_text)

    # Lemmatize and remove punctuations
    # punctuations = list(string.punctuation)
    # preprocessed_text = ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in punctuations])

    preprocessed_text = ' '.join([w for w in tokens])

    # Clear the HTML tags
    # CLEANR = re.compile('<.*?>')
    # preprocessed_text = re.sub(CLEANR, '', preprocessed_text)
    # End your code

    return preprocessed_text

if __name__ == '__main__':
    text = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
    print(f"text: {text}")
    lowercase_text = text.lower()
    print(f"lowercase_text: {lowercase_text}")
    processed_text = remove_stopwords(lowercase_text)
    print(f"removed_stopword_text: {processed_text}")
    punctuations = list(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(processed_text)
    processed_text = ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in punctuations])
    print(f"lemmatized_and_removed_punctuation_text: {processed_text}")
    CLEANR = re.compile('<.*?>')
    processed_text = re.sub(CLEANR, '', processed_text)
    print(f"removed_HTML_tags_text: {processed_text}")