def test_make_windowed_data():
    sentences = [[[1,1], [2,0], [3,3]]]
    sentence_labels = [[1, 2, 3]]
    data = zip(sentences, sentence_labels)
    w_data = make_windowed_data(data, start=[5,0], end=[6,0], window_size=1)

    assert len(w_data) == sum(len(sentence) for sentence in sentences)

    assert w_data == [
        ([5,0] + [1,1] + [2,0], 1,),
        ([1,1] + [2,0] + [3,3], 2,),
        ([2,0] + [3,3] + [6,0], 3,),

        ]

def make_windowed_data(data, start, end, window_size = 1):
    windowed_data = []
    for sentence, labels in data:
        ### YOUR CODE HERE (5-20 lines)
        s_ = [start]*window_size + sentence + [end]*window_size
        for wid in range(len(sentence)):
            a = (sum(s_[wid:(wid+2*window_size+1)],[]), labels[wid])
            windowed_data.append(a)
        ### END YOUR CODE
    return windowed_data
if __name__ == '__main__':
    test_make_windowed_data()