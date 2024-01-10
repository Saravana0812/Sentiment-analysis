import mysql.connector
import datetime
#establishing the connection
def logging(data, res):
    conn = mysql.connector.connect(user='root', password='Drise$6', host='127.0.0.1',auth_plugin='mysql_native_password',database='sentiment_analysis_db')
    now_t= datetime.datetime.now()
    formatted_date = now_t.strftime('%Y-%m-%d %H:%M:%S')
    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    sql = """INSERT INTO Senti_analysis (Processed_at, Content, Result) VALUES (%s,%s, %s);"""
    print(data)
    record = (formatted_date, data, res)
    cursor.execute(sql,record)
    conn.commit()
    #Closing the connection
    conn.close()