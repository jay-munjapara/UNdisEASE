

from app.home import blueprint
from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound
import pickle
import pandas as pd
import numpy as np


def return_top_n_pred_prob_df(n, model, X_test, column_name):
    predictions = model.predict_proba(X_test)
    preds_idx = np.argsort(-predictions)
    classes = pd.DataFrame(model.classes_, columns=['class_name'])
    classes.reset_index(inplace=True)
    top_n_preds = pd.DataFrame()
    for i in range(n):
        top_n_preds[column_name + '_prediction_{}_num'.format(
            i)] = [preds_idx[doc][i] for doc in range(len(X_test))]
        top_n_preds[column_name + '_prediction_{}_probability'.format(
            i)] = [predictions[doc][preds_idx[doc][i]] for doc in range(len(X_test))]
        top_n_preds = top_n_preds.merge(
            classes, how='left', left_on=column_name + '_prediction_{}_num'.format(i), right_on='index')
        top_n_preds = top_n_preds.rename(
            columns={'class_name': column_name + '_prediction_{}'.format(i)})
        try:
            top_n_preds.drop(
                columns=['index', column_name + '_prediction_{}_num'.format(i)], inplace=True)
        except:
            pass
    return top_n_preds


syptoms_lst = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
               'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose']
data_set = [['Drug Reaction', 'An adverse drug reaction (ADR) is an injury caused by taking medication. ADRs may occur following a single dose or prolonged administration of a drug or result from the combination of two or more drugs.'], ['Malaria', 'An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito or by a contaminated needle or transfusion. Falciparum malaria is the most deadly type.'], ['Allergy', "An allergy is an immune system response to a foreign substance that's not typically harmful to your body.They can include certain foods, pollen, or pet dander. Your immune system's job is to keep you healthy by fighting harmful pathogens."], ['Hypothyroidism', 'Hypothyroidism, also called underactive thyroid or low thyroid, is a disorder of the endocrine system in which the thyroid gland does not produce enough thyroid hormone.'], ['Psoriasis', "Psoriasis is a common skin disorder that forms thick, red, bumpy patches covered with silvery scales. They can pop up anywhere, but most appear on the scalp, elbows, knees, and lower back. Psoriasis can't be passed from person to person. It does sometimes happen in members of the same family."], ['GERD', 'Gastroesophageal reflux disease, or GERD, is a digestive disorder that affects the lower esophageal sphincter (LES), the ring of muscle between the esophagus and stomach. Many people, including pregnant women, suffer from heartburn or acid indigestion caused by GERD.'], ['Chronic cholestasis', 'Chronic cholestatic diseases, whether occurring in infancy, childhood or adulthood, are characterized by defective bile acid transport from the liver to the intestine, which is caused by primary damage to the biliary epithelium in most cases'], ['hepatitis A', "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. The virus is one of several types of hepatitis viruses that cause inflammation and affect your liver's ability to function."], ['Osteoarthristis', 'Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of your bones wears down over time.'], ['(vertigo) Paroymsal  Positional Vertigo', "Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — the sudden sensation that you're spinning or that the inside of your head is spinning. Benign paroxysmal positional vertigo causes brief episodes of mild to intense dizziness."], ['Hypoglycemia', " Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. Glucose is your body's main energy source. Hypoglycemia is often related to diabetes treatment. But other drugs and a variety of conditions — many rare — can cause low blood sugar in people who don't have diabetes."], ['Acne', 'Acne vulgaris is the formation of comedones, papules, pustules, nodules, and/or cysts as a result of obstruction and inflammation of pilosebaceous units (hair follicles and their accompanying sebaceous gland). Acne develops on the face and upper trunk. It most often affects adolescents.'], ['Diabetes', 'Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.'], ['Impetigo', "Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects infants and children. Impetigo usually appears as red sores on the face, especially around a child's nose and mouth, and on hands and feet. The sores burst and develop honey-colored crusts."], ['Hypertension', 'Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms.'], ['Peptic ulcer diseae', 'Peptic ulcer disease (PUD) is a break in the inner lining of the stomach, the first part of the small intestine, or sometimes the lower esophagus. An ulcer in the stomach is called a gastric ulcer, while one in the first part of the intestines is a duodenal ulcer.'], ['Dimorphic hemorrhoids(piles)', 'Hemorrhoids, also spelled haemorrhoids, are vascular structures in the anal canal. In their ... Other names, Haemorrhoids, piles, hemorrhoidal disease .'], ['Common Cold', "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold."], ['Chicken pox', 'Chickenpox is a highly contagious disease caused by the varicella-zoster virus (VZV). It can cause an itchy, blister-like rash. The rash first appears on the chest, back, and face, and then spreads over the entire body, causing between 250 and 500 itchy blisters.'], ['Cervical spondylosis', 'Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs).'], ['Hyperthyroidism', "Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the hormone thyroxine. Hyperthyroidism can accelerate your body's metabolism, causing unintentional weight loss and a rapid or irregular heartbeat."], ['Urinary tract infection',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'Urinary tract infection: An infection of the kidney, ureter, bladder, or urethra. Abbreviated UTI. Not everyone with a UTI has symptoms, but common symptoms include a frequent urge to urinate and pain or burning when urinating.'], ['Varicose veins', 'A vein that has enlarged and twisted, often appearing as a bulging, blue blood vessel that is clearly visible through the skin. Varicose veins are most common in older adults, particularly women, and occur especially on the legs.'], ['AIDS', "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). By damaging your immune system, HIV interferes with your body's ability to fight infection and disease."], ['Paralysis (brain hemorrhage)', 'Intracerebral hemorrhage (ICH) is when blood suddenly bursts into brain tissue, causing damage to your brain. Symptoms usually appear suddenly during ICH. They include headache, weakness, confusion, and paralysis, particularly on one side of your body.'], ['Typhoid', 'An acute illness characterized by fever caused by infection with the bacterium Salmonella typhi. Typhoid fever has an insidious onset, with fever, headache, constipation, malaise, chills, and muscle pain. Diarrhea is uncommon, and vomiting is not usually severe.'], ['Hepatitis B', "Hepatitis B is an infection of your liver. It can cause scarring of the organ, liver failure, and cancer. It can be fatal if it isn't treated. It's spread when people come in contact with the blood, open sores, or body fluids of someone who has the hepatitis B virus."], ['Fungal infection', 'In humans, fungal infections occur when an invading fungus takes over an area of the body and is too much for the immune system to handle. Fungi can live in the air, soil, water, and plants. There are also some fungi that live naturally in the human body. Like many microbes, there are helpful fungi and harmful fungi.'], ['Hepatitis C', 'Inflammation of the liver due to the hepatitis C virus (HCV), which is usually spread via blood transfusion (rare), hemodialysis, and needle sticks. The damage hepatitis C does to the liver can lead to cirrhosis and its complications as well as cancer.'], ['Migraine', "A migraine can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities."], ['Bronchial Asthma', 'Bronchial asthma is a medical condition which causes the airway path of the lungs to swell and narrow. Due to this swelling, the air path produces excess mucus making it hard to breathe, which results in coughing, short breath, and wheezing. The disease is chronic and interferes with daily working.'], ['Alcoholic hepatitis', "Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over an extended period of time. It's also aggravated by binge drinking and ongoing alcohol use. If you develop this condition, you must stop drinking alcohol"], ['Jaundice', 'Yellow staining of the skin and sclerae (the whites of the eyes) by abnormally high blood levels of the bile pigment bilirubin. The yellowing extends to other tissues and body fluids. Jaundice was once called the "morbus regius" (the regal disease) in the belief that only the touch of a king could cure it'], ['Hepatitis E', 'A rare form of liver inflammation caused by infection with the hepatitis E virus (HEV). It is transmitted via food or drink handled by an infected person or through infected water supplies in areas where fecal matter may get into the water. Hepatitis E does not cause chronic liver disease.'], ['Dengue', 'an acute infectious disease caused by a flavivirus (species Dengue virus of the genus Flavivirus), transmitted by aedes mosquitoes, and characterized by headache, severe joint pain, and a rash. — called also breakbone fever, dengue fever.'], ['Hepatitis D', 'Hepatitis D, also known as the hepatitis delta virus, is an infection that causes the liver to become inflamed. This swelling can impair liver function and cause long-term liver problems, including liver scarring and cancer. The condition is caused by the hepatitis D virus (HDV).'], ['Heart attack', 'The death of heart muscle due to the loss of blood supply. The loss of blood supply is usually caused by a complete blockage of a coronary artery, one of the arteries that supplies blood to the heart muscle.'], ['Pneumonia', 'Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe.'], ['Arthritis', 'Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis.'], ['Gastroenteritis', 'Gastroenteritis is an inflammation of the digestive tract, particularly the stomach, and large and small intestines. Viral and bacterial gastroenteritis are intestinal infections associated with symptoms of diarrhea , abdominal cramps, nausea , and vomiting .'], ['Tuberculosis', 'Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis (MTB) bacteria. Tuberculosis generally affects the lungs, but can also affect other parts of the body. Most infections show no symptoms, in which case it is known as latent tuberculosis.']]
data_set1 = [['Drug Reaction', 'stop irritation', 'consult nearest hospital', 'stop taking drug', 'follow up'], ['Malaria', 'Consult nearest hospital', 'avoid oily food', 'avoid non veg food', 'keep mosquitos out'], ['Allergy', 'apply calamine', 'cover area with bandage', "-", 'use ice to compress itching'], ['Hypothyroidism', 'reduce stress', 'exercise', 'eat healthy', 'get proper sleep'], ['Psoriasis', 'wash hands with warm soapy water', 'stop bleeding using pressure', 'consult doctor', 'salt baths'], ['GERD', 'avoid fatty spicy food', 'avoid lying down after eating', 'maintain healthy weight', 'exercise'], ['Chronic cholestasis', 'cold baths', 'anti itch medicine', 'consult doctor', 'eat healthy'], ['hepatitis A', 'Consult nearest hospital', 'wash hands through', 'avoid fatty spicy food', 'medication'], ['Osteoarthristis', 'acetaminophen', 'consult nearest hospital', 'follow up', 'salt baths'], ['(vertigo) Paroymsal  Positional Vertigo', 'lie down', 'avoid sudden change in body', 'avoid abrupt head movment', 'relax'], ['Hypoglycemia', 'lie down on side', 'check in pulse', 'drink sugary drinks', 'consult doctor'], ['Acne', 'bath twice', 'avoid fatty spicy food', 'drink plenty of water', 'avoid too many products'], ['Diabetes ', 'have balanced diet', 'exercise', 'consult doctor', 'follow up'], ['Impetigo', 'soak affected area in warm water', 'use antibiotics', 'remove scabs with wet compressed cloth', 'consult doctor'], ['Hypertension ', 'meditation', 'salt baths', 'reduce stress', 'get proper sleep'], ['Peptic ulcer diseae', 'avoid fatty spicy food', 'consume probiotic food', 'eliminate milk', 'limit alcohol'], ['Dimorphic hemmorhoids(piles)', 'avoid fatty spicy food', 'consume witch hazel', 'warm bath with epsom salt', 'consume alovera juice'], ['Common Cold', 'drink vitamin c rich drinks', 'take vapour', 'avoid cold food', 'keep fever in check'], ['Chicken pox', 'use neem in bathing ', 'consume neem leaves', 'take vaccine', 'avoid public places'], ['Cervical spondylosis', 'use heating pad or cold pack', 'exercise', 'take otc pain reliver', 'consult doctor'], [
    'Hyperthyroidism', 'eat healthy', 'massage', 'use lemon balm', 'take radioactive iodine treatment'], ['Urinary tract infection', 'drink plenty of water', 'increase vitamin c intake', 'drink cranberry juice', 'take probiotics'], ['Varicose veins', 'lie down flat and raise the leg high', 'use oinments', 'use vein compression', 'dont stand still for long'], ['AIDS', 'avoid open cuts', 'wear ppe if possible', 'consult doctor', 'follow up'], ['Paralysis (brain hemorrhage)', 'massage', 'eat healthy', 'exercise', 'consult doctor'], ['Typhoid', 'eat high calorie vegitables', 'antiboitic therapy', 'consult doctor', 'medication'], ['Hepatitis B', 'consult nearest hospital', 'vaccination', 'eat healthy', 'medication'], ['Fungal infection', 'bath twice', 'use detol or neem in bathing water', 'keep infected area dry', 'use clean cloths'], ['Hepatitis C', 'Consult nearest hospital', 'vaccination', 'eat healthy', 'medication'], ['Migraine', 'meditation', 'reduce stress', 'use poloroid glasses in sun', 'consult doctor'], ['Bronchial Asthma', 'switch to loose cloothing', 'take deep breaths', 'get away from trigger', 'seek help'], ['Alcoholic hepatitis', 'stop alcohol consumption', 'consult doctor', 'medication', 'follow up'], ['Jaundice', 'drink plenty of water', 'consume milk thistle', 'eat fruits and high fiberous food', 'medication'], ['Hepatitis E', 'stop alcohol consumption', 'rest', 'consult doctor', 'medication'], ['Dengue', 'drink papaya leaf juice', 'avoid fatty spicy food', 'keep mosquitos away', 'keep hydrated'], ['Hepatitis D', 'consult doctor', 'medication', 'eat healthy', 'follow up'], ['Heart attack', 'call ambulance', 'chew or swallow asprin', 'keep calm', "-"], ['Pneumonia', 'consult doctor', 'medication', 'rest', 'follow up'], ['Arthritis', 'exercise', 'use hot and cold therapy', 'try acupuncture', 'massage'], ['Gastroenteritis', 'stop eating solid food for while', 'try taking small sips of water', 'rest', 'ease back into eating'], ['Tuberculosis', 'cover mouth', 'consult doctor', 'medication', 'rest']]
symptoms = set()


@blueprint.route('/index')
@login_required
def index():
    return render_template('index.html', syptoms_lst=syptoms_lst, segment='index')

@blueprint.route('/diagnose')
def diagnose():
    return render_template('diagnosis.html', syptoms_lst=syptoms_lst, segment='diagnosis')

@blueprint.route('/add', methods=['GET', 'POST'])
def add():
    if request.form:
        sym = request.form['symptom']
        symptoms.add(sym)
        return render_template('diagnosis.html', syptoms_lst=syptoms_lst, symptoms=symptoms, segment='sign-up')
    return render_template('diagnosis.html', syptoms_lst=syptoms_lst,  segment='diagnosis')


@blueprint.route('/clear', methods=['GET', 'POST'])
def clear():
    symptoms.clear()
    return render_template('diagnosis.html', syptoms_lst=syptoms_lst, symptoms=symptoms, segment='diagnosis')


@blueprint.route('/clear_symptome', methods=['GET', 'POST'])
def clear_symptome():
    if request.form:
        symptoms.remove(request.form['value'][:-4])
        return render_template('diagnosis.html', syptoms_lst=syptoms_lst, symptoms=symptoms, segment='diagnosis')


@blueprint.route('/give_disease', methods=['GET', 'POST'])
def give_disease():
    diseases = []
    test = np.zeros([1, 131], dtype=int)
    for i in symptoms:
        test[0][syptoms_lst.index(i)] = 1
    loaded_model = pickle.load(open(r'C:\Users\Jay Munjapara\Projects\EVATHON\UndisEASE\xg1.sav', 'rb'))
    result = return_top_n_pred_prob_df(3, loaded_model, test, "test")
    result = result.values.tolist()
    desc = []
    precaution = []
    for i in data_set:
        if i[0] == result[0][1]:
            desc.append(i)
    for i in data_set:
        if i[0] == result[0][3]:
            desc.append(i)
    for i in data_set:
        if i[0] == result[0][5]:
            desc.append(i)
            
    for i in data_set1:
        if i[0] == result[0][1]:
            precaution.append(i)
    for i in data_set1:
        if i[0] == result[0][3]:
            precaution.append(i)
    for i in data_set1:
        if i[0] == result[0][5]:
            precaution.append(i)
    return render_template('diagnosis.html', result=result, precaution=precaution, desc=desc, syptoms_lst=syptoms_lst, symptoms=symptoms, segment='diagnosis')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template(template, segment=segment)

    except TemplateNotFound:
        return render_template('page-404.html'), 404

    except:
        return render_template('page-500.html'), 500

# Helper - Extract current page name from request


def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
