import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as pre






data = pd.read_csv("survey_results_public.csv")


# Title and Sidebar Title
st.title("2023 Developer Salary Web App")
st.sidebar.title("2023 Developer Salary Web App")

# Radio button for user selection
app_mode = st.sidebar.radio("Select App Mode", ["Salary Analysis", "Predict Salary"])


def user_input_features():
    Age= st.sidebar.selectbox("What is your age?",['Under 18 years old','18-24 years old', '25-34 years old','35-44 years old', '45-54 years old','55-64 years old',
       '65 years or older', 'Prefer not to say'],key="age")
    Employment= st.sidebar.multiselect("Which of the following best describes your current employment status? Select all that apply",
                                        ["Employed, full-time", "Employed, part-time", "Independent contractor, freelancer, or self-employed", 
                                         "Retired", "I prefer not to say"], key="employment")
    RemoteWork = st.sidebar.selectbox("Which best describes your current work situation?",['Remote', 'Hybrid (some remote, some in-person)', 'In-person'],key="worksituation")
    EdLevel = st.sidebar.selectbox("Which of the following best describes the highest level of formal education that you have completed?",
                                   ['Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
                                    'Some college/university study without earning a degree',
                                    'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
                                    'Primary/elementary school',
                                    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)',
                                    'Associate degree (A.A., A.S., etc.)',
                                    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
                                    'Something else'], key="Edulevel")
    LearnCode = st.sidebar.multiselect("How do you learn to code? Select all that apply.",
                                       ['Friend or family member', 'School (i.e., University, College, etc)', 'Coding Bootcamp', 
                                        'Colleague', 'Friend or family member', 'Other online resources (e.g., videos, blogs, forum)',
                                        'Books / Physical media', 'Online Courses or Certification', 'Hackathons (virtual or in-person)',
                                        'On the job training'], key="learncode")
    YearsCodePro = st.sidebar.slider("NOT including education, how many years have you coded professionally (as a part of your work)?",
                                     min_value=None, max_value=70, value=0, step=1, key="yearcode")
    Industry=st.sidebar.selectbox("What industry is the company you work for in?",['Information Services, IT, Software Development, or other Technology',
       'Other', 'Financial Services',
       'Manufacturing, Transportation, or Supply Chain',
       'Retail and Consumer Services', 'Higher Education',
       'Legal Services', 'Insurance', 'Healthcare', 'Oil & Gas',
       'Wholesale', 'Advertising Services'],key="industry")
    DevType= st.sidebar.selectbox("Which of the following describes your current job, the one you do most of the time?",['Senior Executive (C-Suite, VP, etc.)', 'Developer, back-end',
       'Developer, front-end', 'Developer, full-stack',
       'System administrator',
       'Developer, desktop or enterprise applications',
       'Developer, QA or test', 'Designer',
       'Data scientist or machine learning specialist',
       'Data or business analyst', 'Security professional', 'Educator',
       'Research & Development role', 'Other (please specify):',
       'Developer, mobile', 'Database administrator',
       'Developer, embedded applications or devices', 'Student',
       'Engineer, data', 'Hardware Engineer', 'Product manager',
       'Academic researcher', 'Developer, game or graphics',
       'Cloud infrastructure engineer', 'Engineering manager',
       'Developer Experience', 'Project manager', 'DevOps specialist',
       'Engineer, site reliability', 'Blockchain', 'Developer Advocate',
       'Scientist', 'Marketing or sales professional'],key="Devtype")
    OrgSize = st.sidebar.selectbox("What is the size of your company",['Just me - I am a freelancer, sole proprietor, etc.','2 to 9 employees', '10 to 19 employees','20 to 99 employees','100 to 499 employees','500 to 999 employees','1,000 to 4,999 employees',  '5,000 to 9,999 employees', '10,000 or more employees', 
              'I don’t know'],key="orgsize")
    Country=st.sidebar.selectbox("Where do you live?",['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Central African Republic', 'Chile', 'China', 'Colombia', 'Congo, Republic of the...', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', "Côte d'Ivoire", "Democratic People's Republic of Korea", 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong (S.A.R.)', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of...', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libyan Arab Jamahiriya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Nomadic', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Republic of Korea', 'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 'Tajikistan', 'Thailand', 'The former Yugoslav Republic of Macedonia', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom of Great Britain and Northern Ireland', 'United Republic of Tanzania', 'United States of America', 'Uruguay', 'Uzbekistan', 'Venezuela, Bolivarian Republic of...', 'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe'],key="country")
    Language= st.sidebar.multiselect("Which programming, scripting, and markup languages have you done extensive development work in over the past year",
                                       ['Ada', 'Apex', 'APL', 'Assembly', 'Bash/Shell (all shells)', 'C ', 'C#', 'C++', 'Clojure', 'Cobol', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Flow', 'Fortran', 'GDScript', 'Go', 'Groovy', 'Haskell', 'HTML/CSS', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua', 'MATLAB', 'Nim', 'Objective-C', 'OCaml', 'Perl', 'PHP', 'PowerShell', 'Prolog', 'Python', 'R ', 'Raku', 'Ruby', 'Rust', 'SAS', 'Scala', 'Solidity', 'SQL', 'Swift', 'TypeScript', 'VBA', 'Visual Basic (.Net)', 'Zig'], key="language")
    features={'Age': Age, 'Employment': Employment, 'RemoteWork': RemoteWork, 'EdLevel': EdLevel , 'LearnCode': LearnCode, 'YearsCodePro': YearsCodePro,'Industry': Industry, 'DevType': DevType, 'OrgSize': OrgSize, 'Country': Country, 'LanguageHaveWorkedWith': Language}
    data_predict=pd.DataFrame(features)
    return data_predict
df=user_input_features()
# Example usage

# Display content based on user selection
if app_mode == "Salary Analysis":
    st.markdown("You've selected Salary Analysis.")
    # Add code for salary analysis here
    
elif app_mode == "Predict Salary":
    st.markdown("You've selected Predict Salary.")
    # Add code for salary prediction here
