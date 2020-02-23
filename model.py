
def rnn_gru_translation_model(
        input_shape, 
        output_sequence_length, 
        english_vocab_size, 
        french_vocab_size,
        learning_rate = 0.01,
        embedding_size=512,
        gru_units=256,
        time_distributed_units=128,
        dropout=.5):
    
    embedding = Embedding(
        input_dim=english_vocab_size, 
        output_dim=embedding_size,  
        input_length=output_sequence_length, 
        input_shape=input_shape[1:])
    
    bidirectional_gru = Bidirectional(GRU(
        units=gru_units,
        return_sequences=True))

    td0 = TimeDistributed(
        layer=Dense(units=french_vocab_size, activation='relu'))

    do0 = Dropout(rate=dropout)
    
    td1 = TimeDistributed(
        layer=Dense(units=french_vocab_size, activation='relu'))

    do1 = Dropout(rate=dropout)

    td2 = TimeDistributed(
        layer=Dense(units=french_vocab_size, activation='softmax'))
    
    model = Sequential([embedding, bidirectional_gru, td0, do0, td1, do1, td2])
    
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=['accuracy'])
    
    return model


