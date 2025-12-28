import os
import json
import time
import tempfile
import psycopg2
import uuid

import streamlit as st
import pandas as pd
from confluent_kafka import Producer

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:9092")


def get_predict_density_distribution(y_test_proba, output):
    plt.figure(figsize=(12, 8))
    plt.hist(np.asarray(y_test_proba), bins=100, density=True, alpha=0.8)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.title("Distribution of predicted density")
    plt.savefig(output + "//density_distribution.png")
    plt.close()


def send_df_to_kafka(
        df,
        topic,
        bootstrap_servers,
        sleep_sec,
        limit_rows):
    producer = Producer({"bootstrap.servers": bootstrap_servers})

    df = df.copy()
    if limit_rows is not None:
        df = df.head(limit_rows)

    df.insert(0, "transaction_id", [str(uuid.uuid4()) for _ in range(len(df))])

    progress_bar = st.progress(0.0)
    total_rows = len(df)

    for idx, row in df.iterrows():
        payload = {
            "transaction_id": row["transaction_id"],
            "data": row.drop("transaction_id").to_dict(),
        }

        producer.produce(topic, value=json.dumps(payload).encode("utf-8"))
        producer.poll(0)

        progress_bar.progress((idx + 1) / max(total_rows, 1))

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    producer.flush()


st.set_page_config(layout="wide")
st.title("Data loader to Kafka")
tab_send, tab_settings, show_results = st.tabs(["Send Files", "Settings", "Show results"])

with tab_settings:
    st.subheader("Emulation Process settings")

    topic = st.text_input("Kafka topic", value=os.getenv("KAFKA_TOPIC", "transactions"))
    ss = st.number_input(
        "Pause between messages(sec)",
        min_value=0.0,
        max_value=2.0,
        value=0.01,
        step=0.01,
    )

    limit_text = st.text_input(
        "Row limit (empty = unlimited)",
        value=""
    )
    limit_rows = None
    try:
        limit_rows = int(limit_text)
        if limit_rows <= 0:
            st.warning("Must be positive number")
    except ValueError:
        st.warning("Enter correct number or no limitation will be applied")

    st.markdown(f"""
    **Current connections**
    - **Brokers:** `{DEFAULT_BROKERS}`
    - **Topic:** `{topic}`""")

with tab_send:
    st.subheader("Load CSV")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    df = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    if df is None:
        st.info("Load CSV file")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Rows", str(df.shape[0]))
    c2.metric("Columns", str(df.shape[1]))

    st.write("Preview:")
    st.dataframe(df.head(5), use_container_width=True)
    st.divider()

    st.write(f"Will be sent: **{len(df) if limit_rows is None else min(len(df), limit_rows)}**")

    if st.button("Send to Kafka", type="primary"):
        with st.spinner("Sending..."):
            send_df_to_kafka(
                df=df,
                topic=topic,
                bootstrap_servers=DEFAULT_BROKERS,
                sleep_sec=float(ss),
                limit_rows=limit_rows,
            )

with show_results:

    do_refresh = st.button("🔄", use_container_width=True)

    refresh_nonce = st.session_state.get("refresh_nonce", 0)
    if do_refresh:
        st.session_state["refresh_nonce"] = refresh_nonce + 1
        st.rerun()

    st.subheader("Scores distribution and recent fraud transactions")

    try:
        conn = psycopg2.connect(
            host="postgres",
            port="5432",
            dbname="fraud_scores",
            user="user",
            password="password"
        )

        fraud_sql = """
            SELECT id, score, fraud_flag, created_at
            FROM transactions
            WHERE fraud_flag = 1
            ORDER BY created_at DESC
            LIMIT 10;
        """
        fraud_df = pd.read_sql(fraud_sql, conn)
        if fraud_df.empty:
            st.info("No fraud_flag = 1 transactions")
        else:
            st.markdown("**Last Fraud transactions**")
            st.dataframe(fraud_df, use_container_width=True, hide_index=True)

        st.divider()
        fraud_sql = """
            SELECT score
            FROM transactions
            ORDER BY created_at DESC
            LIMIT 100;
        """
        df = pd.read_sql(fraud_sql, conn)

        if df.empty:
            st.info("No transactions")
        else:
            st.markdown("**Last scores distribution**")
            scores = df["score"].astype(float).to_numpy()

            with tempfile.TemporaryDirectory() as tmpdir:
                get_predict_density_distribution(scores, tmpdir)
                st.image(
                    os.path.join(tmpdir, "density_distribution.png"),
                    caption="Distribution of predicted density",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error. Retry later")

    finally:
        try:
            conn.close()
        except Exception:
            pass
