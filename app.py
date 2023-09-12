import streamlit as st
import pickle
import numpy as np

with open('Models/LogisticRegression_model.pkl', 'rb') as lg_file:
    lg = pickle.load(lg_file)
with open('Models/KNeighborsClassifier_model.pkl', 'rb') as KNN_file:
    knn = pickle.load(KNN_file)
with open('Models/RandomForestClassifier_model.pkl', 'rb') as rf_file:
    rf = pickle.load(rf_file)
with open('Models/GradientBoostingClassifier_model.pkl', 'rb') as gb_file:
    gb = pickle.load(gb_file)

input_features = ['duration', 'service', 'flag', 'src_bytes', 'num_failed_logins',
                  'count', 'rerror_rate', 'srv_rerror_rate', 'diff_srv_rate',
                  'dst_host_count', 'dst_host_diff_srv_rate',
                  'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

clfs = {
    'LogisticRegression': lg,
    'KNeighborsClassifier': knn,
    'RandomForestClassifier': rf,
    'GradientBoostingClassifier': gb,
}


def main():
    st.set_page_config(page_title="Cyber Security Intrusion Detection", page_icon='img_2.png')
    st.image("img.png", caption="DLithe Consultancy Services Pvt Ltd", width=350)
    st.markdown("<h1 style='font-size: 26px;'>Machine Learning Internship Project</h1>", unsafe_allow_html=True)
    st.title('Cyber Security Intrusion Detection Model')
    st.markdown(
        'Paper [Click Here](https://drive.google.com/file/d/1qRdfyRDMoeyAPNQ8H_SHT-jIUdV2eH9l/view?usp=drive_link).')
    st.sidebar.header('Input Data')
    input_values = {}
    for feature in input_features:
        input_values[feature] = st.sidebar.number_input(f'Enter {feature}')
    if st.sidebar.button('Predict'):
        input_data = np.array([input_values[feature] for feature in input_features]).reshape(1, -1)
        predicted_values = {}
        for clf_name, clf_model in clfs.items():
            predicted = clf_model.predict(input_data)
            predicted_values[clf_name] = predicted
        st.subheader('Predictions From Different Models:')
        for clf_name, predicted in predicted_values.items():
            if predicted == 1:
                st.write(f"Predicted values from {clf_name}:", 'Anomaly', ":warning:")
            else:
                st.write(f"Predicted values from {clf_name}:", 'Normal', ":white_check_mark:")
    st.markdown('## About Input')
    st.info('Make sure to provide valid input values for accurate predictions.')
    st.markdown(
        'After Recursive Feature Elimination (RFE) with a RandomForestClassifier Selected Features Are As Follow:')
    st.markdown('''
|Sample No|duration|protocol\_type|service|flag|src\_bytes|land|wrong\_fragment|urgent|num\_failed\_logins|num\_outbound\_cmds|is\_host\_login|count|srv\_count|serror\_rate|srv\_serror\_rate|rerror\_rate|srv\_rerror\_rate|diff\_srv\_rate|dst\_host\_count|dst\_host\_diff\_srv\_rate|Class(Sample)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|0|tcp|private|S0|0|0|0|0|0|0|0|128|18|1\.0|1\.0|0\.0|0\.0|0\.05|255|0\.06|Normal
|2|0|tcp|daytime|S0|0|0|0|0|0|0|0|271|11|1\.0|1\.0|0\.0|0\.0|0\.06|255|0\.07|Normal
|3|0|tcp|ftp\_data|SF|1874|0|0|0|0|0|0|11|11|0\.0|0\.0|0\.0|0\.0|0\.0|255|0\.03|Anomaly
|4|0|tcp|http|SF|235|0|0|0|0|0|0|7|7|0\.0|0\.0|0\.0|0\.0|0\.0|255|0\.01|Normal
|5|0|tcp|ftp|S0|0|0|0|0|0|0|0|250|13|1\.0|1\.0|0\.0|0\.0|0\.06|255|0\.07|Anomaly
|6|0|udp|private|SF|1|0|0|0|0|0|0|55|9|0\.0|0\.0|0\.0|0\.0|0\.07|255|0\.23|Normal
|7|0|tcp|http|SF|303|0|0|0|0|0|0|2|2|0\.0|0\.0|0\.0|0\.0|0\.0|255|0\.01|Anomaly
|8|0|tcp|ftp|RSTO|0|0|0|0|0|0|0|266|6|0\.0|0\.0|1\.0|1\.0|0\.06|255|0\.07|Normal
|9|2035|udp|other|SF|147|0|0|0|0|0|0|1|1|0\.0|0\.0|0\.0|0\.0|0\.0|255|0\.62|Anomaly
|10|9|udp|private|SF|105|0|0|0|0|0|0|2|1|0\.0|0\.0|0\.0|0\.0|1\.0|255|0\.01|Normal
        ''')
    st.image("img_1.png", caption="Training Results", use_column_width=True)
    st.markdown('## About Dataset')
    st.markdown('For Dataset [Click Here](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection).')
    st.markdown('''
    1. `duration` : This column represents the duration of the connection in seconds, indicating how long a network connection was active during the observation.

    2. `protocol_type` : This column contains the type of network protocol used in the connection, such as TCP, UDP, or ICMP.

    3. `service` : The "service" column specifies the network service or application associated with the connection, like HTTP, FTP, or SSH.

    4. `flag` : The "flag" column indicates the status or state of the network connection, often including values like SYN, ACK, or FIN, which represent different phases of a TCP connection.

    5. `src_bytes` : This column represents the number of bytes sent from the source to the destination in the network connection.

    6. `dst_bytes` : Similar to `src_bytes`, this column represents the number of bytes sent from the destination to the source in the network connection.

    7. `land` : A binary column indicating whether the connection is from/to the same host (1 if true, 0 if false). Land attacks involve spoofing the source and destination addresses to make it appear as if a connection is originating and ending at the same host.

    8. `wrong_fragment` : This column counts the number of wrong fragments in the network packet or connection, which can be indicative of certain types of attacks.

    9. `urgent` : The "urgent" column represents the number of urgent packets in the connection, where urgent packets typically request immediate processing.

    10. `hot` : This column counts the number of "hot" indicators in the connection, which may relate to suspicious or unusual activity.

    11. `num_failed_logins` : It records the number of failed login attempts during the connection, which can indicate potential unauthorized access attempts.

    12. `logged_in` : A binary column indicating whether the user was logged in during the connection (1 if true, 0 if false).

    13. `num_compromised` : This column represents the number of compromised conditions detected during the connection, which could suggest a security breach.

    14. `root_shell` : A binary column indicating whether a root shell was obtained during the connection (1 if true, 0 if false). Gaining root access is a significant security concern.

    15. `su_attempted` : A binary column indicating whether an "su" (superuser) command was attempted during the connection (1 if true, 0 if false).

    16. `num_root` : This column counts the number of "root" accesses during the connection, indicating potentially unauthorized attempts to gain root access.

    17. `num_file_creations` : The number of file creation operations during the connection, which could be relevant in detecting suspicious activity like malware propagation.

    18. `num_shells` : This column counts the number of shell prompts opened during the connection, which can be indicative of malicious activity.

    19. `num_access_files` : Represents the number of access file operations during the connection, which can be relevant in security analysis.

    20. `num_outbound_cmds` : This column typically contains zeros and may not provide significant information since it counts the number of outbound commands in the connection.

    21. `is_host_login` : A binary column indicating whether the login is associated with a host login scenario (1 if true, 0 if false).

    22. `is_guest_login` : A binary column indicating whether the login is associated with a guest login scenario (1 if true, 0 if false).

    23. `count` : This column represents the number of connections to the same host as the current connection in the last 2 seconds.

    24. `srv_count` : Similar to `count`, but specific to services, indicating the number of connections to the same service as the current connection in the last 2 seconds.

    25. `serror_rate` : The percentage of connections that have "SYN" errors, indicating connection errors on the client side.

    26. `srv_serror_rate` : Similar to `serror_rate`, but specific to services.

    27. `rerror_rate` : The percentage of connections that have "REJ" errors, indicating rejected connections.

    28. `srv_rerror_rate` : Similar to `rerror_rate`, but specific to services.

    29. `same_srv_rate` : The percentage of connections to the same service as the current connection among all connections.

    30. `diff_srv_rate` : The percentage of connections to different services compared to all connections.

    31. `srv_diff_host_rate` : The percentage of connections to different hosts compared to connections to the same service.

    32. `dst_host_count` : The number of connections to the same destination host as the current connection.

    33. `dst_host_srv_count` : The number of connections to the same destination host using the same service as the current connection.

    34. `dst_host_same_srv_rate` : The percentage of connections to the same service on the destination host among all connections to that host.

    35. `dst_host_diff_srv_rate` : The percentage of connections to different services on the destination host among all connections to that host.

    36. `dst_host_same_src_port_rate` : The percentage of connections from the same source port among all connections to the same destination host.

    37. `dst_host_srv_diff_host_rate` : The percentage of connections to different hosts using the same service on the destination host.

    38. `dst_host_serror_rate` : The percentage of connections that have "SYN" errors on the destination host.

    39. `dst_host_srv_serror_rate` : Similar to `dst_host_serror_rate`, but specific to services on the destination host.

    40. `dst_host_rerror_rate` : The percentage of connections that have "REJ" errors on the destination host.

    41. `dst_host_srv_rerror_rate` : Similar to `dst_host_rerror_rate`, but specific to services on the destination host.

    42. `class` : This column contains the label or class of each network connection, indicating whether it is a normal or anomaly connection.''')


if __name__ == '__main__':
    main()
