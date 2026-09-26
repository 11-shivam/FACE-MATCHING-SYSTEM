import face_embedding
import database_connection
import cv2
import json
import numpy as np


# ----------------------------------------
# Capture face from webcam
# ----------------------------------------
def capture_face():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not working !!")
        return None

    print("Press 's' to capture face")
    print("Press 'q' to quit")

    frame = None

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera not working !!")
            frame = None
            break

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        cv2.imshow("Face Matching", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            frame = None
            break

        if key == ord('s'):
            print("Face captured!")
            break

    camera.release()
    cv2.destroyAllWindows()

    return frame


# ----------------------------------------
# Get people from database
# ----------------------------------------
def get_database():

    connector = database_connection.get_connection()

    if connector is None:
        print("Database connection failed!")
        return []

    cursor = connector.cursor()

    query = """
        SELECT id, full_name, face_embedding
        FROM people
    """

    try:

        cursor.execute(query)

        data = cursor.fetchall()

        return data

    except Exception as e:

        print("Database error:", e)
        return []

    finally:

        cursor.close()
        connector.close()


# ----------------------------------------
# Calculate cosine distance
# ----------------------------------------
def cosine_distance(a, b):

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # Avoid division by zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return float("inf")

    similarity = np.dot(a, b) / (norm_a * norm_b)

    distance = 1 - similarity

    return distance


# ----------------------------------------
# Find closest person
# ----------------------------------------
def find_match(camera_embedding):

    people = get_database()

    if not people:
        print("No people found in database.")
        return None, float("inf")

    best_person = None
    best_distance = float("inf")

    for person in people:

        # person:
        # person[0] -> id
        # person[1] -> full_name
        # person[2] -> face_embedding

        try:

            stored_embedding = json.loads(person[2])

        except Exception as e:

            print(
                f"Could not read embedding for {person[1]}:",
                e
            )

            continue

        distance = cosine_distance(
            camera_embedding,
            stored_embedding
        )

        # print(f"{person[1]} -> distance: {distance:.4f}")

        if distance < best_distance:

            best_distance = distance
            best_person = person

    return best_person, best_distance


# ----------------------------------------
# Main
# ----------------------------------------

THRESHOLD = 0.4

face = capture_face()

if face is not None:

    # Generate embedding of captured face
    camera_embedding = face_embedding.generate_embedding(face)

    if camera_embedding is None:

        print("Could not generate face embedding.")

    else:

        print("Face embedding generated.")

        # Search database
        person, distance = find_match(camera_embedding)

        print("\n-----------------------------")
        print("Best distance:", distance)
        print("-----------------------------")

        if person is not None and distance < THRESHOLD:

            print("MATCH FOUND")
            print("Person ID:", person[0])
            print("Person:", person[1])
            print("Distance:", distance)

        else:

            print("UNKNOWN FACE")