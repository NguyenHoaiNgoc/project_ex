from sklearn.datasets import load_iris
import streamlit
import pickle
import numpy as np

iris_data = load_iris()
model = pickle.load(open('iris_lrg_model01.pkl', 'rb'))

#tạo thanh ngang 
streamlit.sidebar.title('Choose the sepal/petal length/width')
sepal_length = streamlit.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = streamlit.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = streamlit.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width = streamlit.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)

sepal_length = np.nan_to_num(sepal_length)
sepal_width = np.nan_to_num(sepal_width)
petal_length = np.nan_to_num(petal_length)
petal_width = np.nan_to_num(petal_width)


#dự đoán kết quả
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
streamlit.write('RESULT:')
streamlit.write(iris_data.target_names[prediction[0]])

# import streamlit as st
# import pickle
# import numpy as np
# from sklearn.datasets import load_iris

# iris = load_iris()

# model = pickle.load(open('iris_model_01.pkl', 'rb'))

# def prediction_explain(prediction):
#     explanation = {
#         0: "Setosa",
#         1: "Versicolor",
#         2: "Virginica"
#     }
#     return explanation.get(prediction, "Unknown")

# st.sidebar.title('Iris Classifier')
# sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
# sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
# petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
# petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)

# sepal_length = np.nan_to_num(sepal_length)
# sepal_width = np.nan_to_num(sepal_width)
# petal_length = np.nan_to_num(petal_length)
# petal_width = np.nan_to_num(petal_width)

# st.write('### Các đặc trưng đã chọn:')
# st.write(f" - Sepal Length: {sepal_length}")
# st.write(f" - Sepal Width: {sepal_width}")
# st.write(f" - Petal Length: {petal_length}")
# st.write(f" - Petal Width: {petal_width}")

# prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# st.write(f'### Dự đoán: {prediction_explain(prediction[0])}')