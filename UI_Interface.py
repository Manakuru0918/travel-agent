import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from travel_agent import agent_executor, booking_manager  

st.set_page_config(page_title="Smart Travel Agent", layout="wide")
st.title("✈️ Smart Travel Agent Chatbot")

#  Moved this to the top (initialize session state safely)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask me about weather, trip planning, or currency conversion...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        output = response["output"]
        st.session_state.chat_history.append(AIMessage(content=output))

# Chat display
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Booking display section
with st.sidebar:
    st.header("📦 Stored Bookings")
    bookings = booking_manager.get_bookings()
    if bookings:
        for i, booking in enumerate(bookings, 1):
            st.write(f"**Booking {i}:**")
            for k, v in booking.items():
                st.write(f"- {k.capitalize()}: {v}")
        total_cost = booking_manager.calculate_total_cost()
        st.success(f"Total Cost: ${total_cost:.2f}")
    else:
        st.info("No bookings added yet.")

