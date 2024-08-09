import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


def sigmoid_function(w, x, b):
    """ Sigmoid function with one feature variable
    """
    return 1/(1+np.exp(-(w*x+b)))

def main():
    st.title("Demo for sigmoid function visualization.")

    w = st.sidebar.slider(label="w", min_value=-10,max_value=10,value=0, step=1)
    b = st.sidebar.slider(label="b", min_value=-10,max_value=10,value=0, step=1)

    # plot the function
    fig, ax = plt.subplots()
    # Define x
    x = np.linspace(start=-20, stop=20, num=100)
    ax.plot(sigmoid_function(w,x,b))
    st.pyplot(fig=fig)

if __name__=="__main__":
    main()