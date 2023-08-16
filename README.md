# GD_Project_01
AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 양주영
- 리뷰어 : 황규빈


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 데이터 전처리 ~ 모델 구성 및 학습까지 완료 했습니다.
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 데이터 전처리에 대한 부분이 좋았습니다. 주석도 친절히 있어서 코드 이해가 쉬웠습니다.
  > 
  > 특히나 인상 깊었던 부분은 데이터 전처리 진행하는데,
  >
  > 특정 전처리 로직을 진행하고 확인하며, 다음 문제를 확인 하는게 너무 좋았습니다.
  >
  > 
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 평가 부분에 살짝 어려움이 있었던걸로 보이나,
  >
  > 고도화를 통해 진행하면 완벽할거 같습니다.
  > 
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  >
  >
  >
```
한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data[:5]
/tmp/ipykernel_3868/3525229848.py:2: FutureWarning: The default value of regex will change from True to False in a future version.
  train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
```
```
#한글이 없는 리뷰는 빈 값이 되었을테니 다시 한번 이상치 제거
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())
```
- [O] 코드가 간결한가요?
  > 네, 
```
# 모델 훈련
from keras.callbacks import ModelCheckpoint
mc = ModelCheckpoint("splstm.h5", monitor='val_accuracy',save_best_only=True)

model_rnn.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

epochs = 20

history_rnn = model_rnn.fit(x_train,
                              y_train,
                              epochs=epochs,
                              callbacks=[mc],
                              batch_size=128,
                              validation_data=(x_val, y_val),
                              verbose=1)
```
