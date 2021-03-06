import DeintelaceVideo
from os import path
import os
import cv2
import pickle
import insightface
import numpy as np
import face_alignment
import Scene_segmentation
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from collections import Counter

# configurations
show = False
ctx = -1 # no GPUs
# alignment and detection
handler = face_alignment.Handler(r'C:\Users\azmihaid\PycharmProjects\LieDet\model\2d106det', 0, ctx_id=ctx,
                                 det_size=640)
# detection model - not in use
detection_model = insightface.model_zoo.get_model('retinaface_r50_v1')
detection_model.prepare(ctx_id=ctx, nms=0.4)
# recognition model
recognition_similarity_thresh = 0.3
recognition_model = insightface.model_zoo.get_model('arcface_r100_v1')
recognition_model.prepare(ctx_id=ctx)


def face_change(last_person, current_person):
    emb1 = last_person.flatten()
    emb2 = current_person.flatten()
    from numpy.linalg import norm
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    if sim > recognition_similarity_thresh:
        return False
    return True


def face_inside_bounds(faces,frame_shape):
    for i in range(len(faces)):
        if faces[i,1] < 0:
            faces[i,1] = 0
        if faces[i,3] > frame_shape[0]:
            faces[i,3] = frame_shape[0]
        if faces[i,0] < 0:
            faces[i,0] = 0
        if faces[i,2] > frame_shape[1]:
            faces[i,2] = frame_shape[1]
    return faces


# face detection ---------------------------------------------------------------------------------
def face_detection(frames,showfigures):
    frames_for_detections = frames.copy()

    # bounding boxes around faces
    faces_bounding_boxes = []
    # landmarks of eyes, mouth, nose
    faces_landmarks = []
    # confidence of detection
    detection_confidence = []
    for fr_idx, fr in enumerate(frames_for_detections):
        #current_found_boxes, current_found_lm = detection_model.detect(fr, threshold=0.5, scale=1.0)
        try:
            current_found_lm, current_found_boxes = handler.get(fr, get_all=True)
            if len(current_found_boxes) > 1:
                raise NameError('More than one face')
        except:
            faces_landmarks.append([])
            faces_bounding_boxes.append([])
            detection_confidence.append([])
            continue
        faces_landmarks.append(np.asarray(current_found_lm, dtype=int))
        faces_bounding_boxes.append(face_inside_bounds(np.asarray(current_found_boxes[:, :4], dtype=int), fr.shape))
        detection_confidence.append(current_found_boxes[:, 4])
        if showfigures:
            for i, box in enumerate(faces_bounding_boxes[fr_idx]):
                cv2.rectangle(fr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            for landmark in faces_landmarks[fr_idx]:
                for l in range(landmark.shape[0]):
                    cv2.circle(fr, (landmark[l][0], landmark[l][1]), 1, (0, 0, 255), 2)
            cv2.imshow('detection',fr)
            cv2.waitKey(1000)
    return faces_bounding_boxes, faces_landmarks, detection_confidence


# face recognition/encoding -----------------------------------------------------------------------
def encode_faces(frames, all_frames_bbs):
    frames_for_encoding = frames.copy()
    face_encodings = []
    for fr_idx, fr in enumerate(frames_for_encoding):
        current_encodings = []
        try:
            frame_bbs = all_frames_bbs[fr_idx]
            for i in range(frame_bbs.shape[0]):
                face_cropped = fr[frame_bbs[i,1]:frame_bbs[i,3],frame_bbs[i,0]:frame_bbs[i,2],:]
                #cv2.imshow('fig',face_cropped)
                #cv2.waitKey(1000)
                encoding = recognition_model.get_embedding(cv2.resize(face_cropped,(112, 112)))
                current_encodings.append(encoding)
        except:
            current_encodings = []
        face_encodings.append(current_encodings)
    return face_encodings


# divide to persons -----------------------------------------------------------------------------------
def cluster_persons_scenes(encodings, scenes_start):
    face_encodings_singles = [np.reshape(x, (512)) for x in encodings if len(x) == 1]
    X = np.asarray(face_encodings_singles)
    kmeans = KMeans(n_clusters=3, max_iter=100*len(encodings), random_state=None).fit(X)
    centroids = kmeans.cluster_centers_
    frame_persons = -1* np.ones(shape=(len(encodings),1))
    start = 0
    scenes_idx_dict = {}
    scene_number = 0
    for idx, face in enumerate(encodings):
        if len(face) != 1:
            frame_persons[idx] = -1
        else:
            frame_persons[idx] = kmeans.predict(face[0])

        if scenes_start[idx] and idx != start:
            # scene is from start to idx
            count_votes = list(frame_persons[start:idx].flatten())
            vote = max(set(count_votes), key=count_votes.count)
            scenes_idx_dict[scene_number] = (vote, [start, idx])
            start = idx
            scene_number +=1

    # combine scenes
    start_idx = 0
    moving_idx = 1
    combined_scenes_dict = {}
    while start_idx < len(scenes_idx_dict.keys())-1:
        combined_scenes_dict[start_idx] = scenes_idx_dict[start_idx]
        person, idxs = scenes_idx_dict[start_idx]
        person2, idxs2 = scenes_idx_dict[moving_idx]
        while person == person2:
            combined_scenes_dict[start_idx] = (person, [idxs[0], idxs2[1]])
            moving_idx +=1
            if moving_idx < len(scenes_idx_dict.keys()):
                person2, idxs2 = scenes_idx_dict[moving_idx]
            else:
                person2 = 'End'
                break
        start_idx = moving_idx
        moving_idx +=1
    return combined_scenes_dict


def track_persons(encodings):
    scenes_starts_persons = np.zeros(shape=(len(encodings),1), dtype=bool)
    for idx, face in enumerate(encodings[:-1]):
        if len(encodings[idx]) == 1 and len(encodings[idx+1]) == 1:
            if face_change(encodings[idx][0],encodings[idx+1][0]):
                scenes_starts_persons[idx+1] = True
        elif len(encodings[idx]) != len(encodings[idx+1]):
            scenes_starts_persons[idx + 1] = True
    return scenes_starts_persons


def save_scenes(folder,frames,scene_cuts):
    num_written_frames = 0
    for scene_index in scene_cuts.keys():
        person, indexes = scene_cuts[scene_index]
        if indexes[1] - indexes[0] < 30:
            continue
        if person == -1:
            continue
        num_written_frames += indexes[1] - indexes[0] + 1
        out = cv2.VideoWriter(path.join(folder,'person' + str(person) + '_scene' + str(scene_index) + '.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 20, (360, 288))
        for f in range(indexes[0],indexes[1]):
            out.write(frames[f])
        out.release()
    return num_written_frames


if __name__ == '__main__':

    file_path = r"C:\Users\azmihaid\OneDrive - Intel Corporation\Desktop\New folder\1.02\VTS_01_4.VOB"
    head_tail = path.split(file_path)
    analysis_folder = path.join(head_tail[0], head_tail[1][:-4] + '_analysis')
    if not path.isdir(analysis_folder):
        os.mkdir(analysis_folder)

    # load frames ---------------------------------------------------------------------------------------
    frames_path = path.join(analysis_folder,'dissected_frames.p')
    if path.isfile(frames_path):
        frames = pickle.load(open(frames_path, "rb"))
    else:
        frames = DeintelaceVideo.dissect_and_deinterlace_video(file_path)
        pickle.dump(frames, open(frames_path, "wb"))

    # scene segmentation based on image structure --------------------------------------------------------
    scene_segmentation_path = path.join(analysis_folder,'scene_segmentation.p')
    if path.isfile(scene_segmentation_path):
        scenes_starts_based_on_structure = pickle.load(open(scene_segmentation_path, "rb"))
    else:
        scenes_starts_based_on_structure = Scene_segmentation.calculate_image_diff(frames)
        pickle.dump(scenes_starts_based_on_structure, open(scene_segmentation_path, "wb"))

    # face detections ------------------------------------------------------------------------------------
    face_detections_path = path.join(analysis_folder,'face_detections.p')
    if path.isfile(face_detections_path):
        bounding_boxes, landmarks, detection_confidences = pickle.load(open(face_detections_path, "rb"))
    else:
        bounding_boxes, landmarks, detection_confidences = face_detection(frames,show)
        to_pickle = (bounding_boxes, landmarks, detection_confidences)
        pickle.dump(to_pickle, open(face_detections_path, "wb"))

    # face recognitions ----------------------------------------------------------------------------------
    # encodings
    face_encodings_path = path.join(analysis_folder,'face_encodings.p')
    if path.isfile(face_encodings_path):
        face_encodings = pickle.load(open(face_encodings_path, "rb"))
    else:
        face_encodings = encode_faces(frames, bounding_boxes)
        pickle.dump(face_encodings, open(face_encodings_path, "wb"))

    # final scene cuts -----------------------------------------------------------------------------------
    scenes_directory = path.join(analysis_folder, 'cut_scenes')
    if not path.exists(scenes_directory):
        scenes_starts_based_on_tracking = track_persons(face_encodings)
        scenes_starts = np.logical_or(scenes_starts_based_on_structure, scenes_starts_based_on_tracking)
        scene_info = cluster_persons_scenes(face_encodings, scenes_starts)
        os.mkdir(scenes_directory)
        frs = save_scenes(scenes_directory,frames,scene_info)
        print('Number of frames saved = ' + str(frs) + ' from ' + str(len(frames)))


    # Gaze detection -------------------------------------------------------------------------------------




    '''

    # save scenes
    out = cv2.VideoWriter(file_path[:-4]+'_scenes.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (360, 288))
    for f in range(len(frames)):
        if scenes_starts[f]:
            for i in range(60):
                out.write(frames[f])
        out.write(frames[f])
    out.release()
    # fineto


# visualization
if faces is not None:
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        # print('score', faces[i][4])
        box = faces[i].astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color,2)
cv2.imshow('Figure',frame)
cv2.waitKey(1)
 '''
'''

def get_box(rect):
    return rect[1], rect[0], rect[3] - rect[1], rect[2] - rect[0]

def get_side_box(rect):
    return rect[1], rect[0] + rect[2], rect[1] + rect[3], rect[0]


def overlap(rect1, rect2):
    # bottom, right, top, left
    # 'low y, high x, high y, low x'
    if rect1[3] >= rect2[1] or rect2[3] >= rect1[1]:
        return False
    if rect1[2] <= rect2[0] or rect2[2] <= rect1[0]:
        return False
    return True

def get_detectors():
    detector = partial(face_recognition.face_locations, model='cnn')
    caspath = path.join(r'C:\Python\Lib\site-packages\cv2\data', r'haarcascade_profileface.xml')
    side = cv2.CascadeClassifier(caspath)

    def side_detector(image):
        return side.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

    return detector, side_detector, face_recognition  # , predictor


def detect_faces(image):
    # Detect faces in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector, side_face_detector, face_recognition = get_detectors()
    face_rects = detector(gray)

    box = face_rects
    for item in map(get_side_box, side_face_detector(image)):
        if len(item):
            box.append(item)
    if len(box) > 1:
        fset = set()
        skip = None
        for a, b in combinations(box, 2):
            a = tuple(a)
            b = tuple(b)
            if skip is a:
                continue
            fset.add(b)
            if overlap(a, b):
                if a in fset:
                    fset.remove(a)
                skip = a
            else:
                fset.add(a)
        box = list(fset)

    # Draw a rectangle around the faces
    if len(box)>0:
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(1)


'''





