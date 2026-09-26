import cv2
import face_embedding
import database_connection
import json
# -------------------------------
# Capture face from camera
# -------------------------------

def capture_face(person_id) :
    camera = cv2.VideoCapture(0)

    print("Press 's' to capture face")
    print("Press 'q' to quit")

    while True :
        success , frame = camera.read()

        if( not success) :
            print("Camera not working !!")
            break

        #Invert the face
        frame = cv2.flip(frame,1)

        cv2.imshow("WebCam",frame)

        key = cv2.waitKey(1) & 0xFF

        if(key  == ord('q')):
            break

        if(key == ord('s')):
            #Issue in this image name , each person image name is Shivam , change it later
            path = f"images/shivam.png"

            cv2.imwrite(path, frame)
            print("Face image saved:", path)
            break

    camera.release()
    cv2.destroyAllWindows()

    return path




# -------------------------------
# Store person in database
# -------------------------------

def database_entry():

    print("---- Database Entry ----")

    person_name = input("Enter Person Name: ").strip()

    if not person_name:
        print("Name cannot be empty!")
        return

    # Capture face
    image_path = capture_face()

    if image_path is None:
        print("No face image captured!")
        return

    # Generate embedding
    embedding = face_embedding.generate_embedding(image_path)

    if embedding is None:
        print("No face detected!")
        return

    print("Face embedding generated!")
    print("Embedding size:", len(embedding))

    # Connect to database
    connection = database_connection.get_connection()

    if connection is None:
        print("Database connection failed!")
        return

    cursor = connection.cursor()

    query = """
        INSERT INTO people
        (
            full_name,
            face_embedding
        )
        VALUES (%s, %s)
    """

    values = (
        person_name,
        json.dumps(embedding)
    )

    try:

        cursor.execute(query, values)

        connection.commit()

        print("Person registered successfully!")

    except Exception as e:

        print("Database error:", e)
        connection.rollback()

    finally:

        cursor.close()
        connection.close()


database_entry()
