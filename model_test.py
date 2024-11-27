import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 평가 메트릭 함수 정의
f1_list = []
bal_list = []


def classifier_eval(y_test, y_pred):
    # 혼동행렬 생성
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print('혼동행렬 : ', cm)

    # 혼동행렬 요소 추출
    TP = cm[1, 1]  # True Positive
    FN = cm[1, 0]  # False Negative
    FP = cm[0, 1]  # False Positive
    TN = cm[0, 0]  # True Negative

    # Precision, Recall 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-Score 계산
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print('F1-Score:', f1)

    # Balance 계산
    PD = recall  # Recall과 동일
    PF = FP / (FP + TN) if (FP + TN) > 0 else 0
    balance = 1 - (((0 - PF) ** 2 + (1 - PD) ** 2) / 2)
    print('Balance :', balance)

    return f1, balance


# Tabula-8B 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("mlfoundations/tabula-8b")
model = AutoModelForCausalLM.from_pretrained("mlfoundations/tabula-8b").to(
    "cuda" if torch.cuda.is_available() else "cpu")


# 입력 텍스트 길이 자르기 함수
def truncate_inputs(input_text, max_length=4096):
    """
    입력 텍스트를 최대 길이로 자릅니다.
    """
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    if tokenized_input['input_ids'].shape[1] > max_length:
        truncated_input = tokenizer.decode(
            tokenized_input['input_ids'][0, :max_length], skip_special_tokens=True
        )
        return truncated_input
    return input_text


# Tabula-8B 추론 함수
def tabula8b_predict(target_example, labeled_examples, target_column, target_choices):
    # JSON 형식으로 입력 준비
    inputs = {
        "target_example": target_example.to_dict(orient="records"),
        "labeled_examples": labeled_examples.to_dict(orient="records"),
        "target_column": target_column,
        "target_choices": target_choices,
    }

    # JSON 데이터를 문자열로 변환
    input_text = str(inputs)

    # 입력 데이터 길이를 자름
    truncated_text = truncate_inputs(input_text, max_length=4096)

    # 모델 입력 준비
    encoded_inputs = tokenizer(truncated_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 추론 수행
    with torch.no_grad():
        outputs = model.generate(encoded_inputs['input_ids'], max_new_tokens=200)

    # 결과 디코딩
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.strip()


# CSV 파일 로드
csv_file_path = 'JDT.csv'
df = pd.read_csv(csv_file_path)

# 특징(X)과 목표 변수(y) 분리
X = df.drop(columns=['class'])
y = df['class']

# 데이터 분할
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-겹 교차 검증 설정
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scaler = MinMaxScaler()
X_test_normalized = scaler.fit_transform(X_test)

# K-겹 교차 검증 수행
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # 데이터 정규화
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # Tabula-8B를 위한 데이터 준비
    labeled_examples = pd.DataFrame(X_fold_train_normalized, columns=X_fold_train.columns)
    labeled_examples["class"] = y_fold_train.reset_index(drop=True)

    target_example = pd.DataFrame(X_test_normalized, columns=X_test.columns).iloc[0:1]  # 테스트 데이터에서 첫 번째 예제 사용

    # Tabula-8B 추론
    target_column = "class"
    target_choices = ["Yes", "No"]
    output = tabula8b_predict(
        target_example=target_example,
        labeled_examples=labeled_examples,
        target_column=target_column,
        target_choices=target_choices
    )

    # 예측 결과 저장
    y_pred = [1 if val == "Yes" else 0 for val in [output]]  # "Yes" => 1, "No" => 0
    y_test_bin = y_test.iloc[0:1].tolist()

    # 평가 지표 계산
    f1, balance = classifier_eval(y_test_bin, y_pred)
    f1_list.append(f1)
    bal_list.append(balance)

# 평균 평가 지표 출력
print('avg_F1-Score: {}'.format((sum(f1_list) / len(f1_list))))
print('avg_Balance: {}'.format((sum(bal_list) / len(bal_list))))
