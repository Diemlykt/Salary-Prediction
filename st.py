# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import altair as alt
from xgboost import XGBRegressor


	
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Others'
    return categorical_map

def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

##def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'##

@st.cache_data
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    select = ['Age', 'EdLevel', 'YearsCodePro', 'DevType', 'Country', 'ConvertedCompYearly']
    sel_df = df[select]
    sel_df = sel_df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    
    ctry_df = sel_df.copy()
    country_map = shorten_categories(ctry_df.Country.value_counts(), 400)
    ctry_df['Country'] = ctry_df['Country'].map(country_map)
    #remove "Others"
    
    
    job_df = ctry_df.copy()
    job_map = shorten_categories(job_df.DevType.value_counts(), 300)
    job_df['DevType'] = job_df['DevType'].map(job_map)
    #remove "Others"
    job_df = job_df[job_df['DevType'] != 'Others']
    
    age_df = job_df.copy()
    age_map = shorten_categories(age_df.Age.value_counts(), 400)
    age_df['Age'] = age_df['Age'].map(age_map)
    #remove "Others"
    age_df = age_df[age_df['Age'] != 'Others']
    
    exp_df = age_df.copy()
    exp_df['YearsCodePro'] = exp_df['YearsCodePro'].apply(clean_experience)
    exp_map = shorten_categories(exp_df.YearsCodePro.value_counts(), 200)
    exp_df['YearsCodePro'] = exp_df['YearsCodePro'].map(exp_map)
    #remove "Others"
    exp_df = exp_df[exp_df['YearsCodePro'] != 'Others']
    
    edu_df = exp_df.copy()
 ##   edu_df['EdLevel'] = edu_df['EdLevel'].apply(clean_education)##
    
    box_df = edu_df.copy()
    final_df = box_df.copy()
    final_df = final_df[final_df["Salary"] <= 300000]
    final_df = final_df[final_df["Salary"] >= 10000]
    
    return final_df


def show_explore():
    st.sidebar.subheader("SURVEY OVERVIEW")   
    selected_chart = st.sidebar.radio("Distribution of correspondences based on: ", ["Country", "Age", "Education Level"])
    st.write(f"#### Distribution of correspondences based on {selected_chart}")
    # Display chart based on user selection
    if selected_chart == "Country":
        data = df["Country"].value_counts().head(10)
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        wedges, texts, autotexts = ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=False, startangle=90)
        ax1.axis("equal")
        for autotext in autotexts:
            autotext.set_fontsize(16)  # Adjust the font size as needed
        # Separate legend from the pie chart with a larger font size
        legend = ax1.legend(wedges, data.index, title="Countries", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop={'size': 16})
        legend.set_title("Countries", prop={'size': 16})
        
        st.pyplot(fig1)

    elif selected_chart == "Age":
        data = df["Age"].value_counts().sort_index()
        fig, ax1 = plt.subplots(figsize=(18, 8))
        ax1.barh(data.index, data, color=plt.cm.Paired(range(len(data))))

        # Customize axis labels and title
        ax1.set_xlabel('Count', fontsize=18)
        ax1.set_ylabel('Age Groups', fontsize=18)
       

        # Increase font size of tick labels
        ax1.tick_params(axis='both', labelsize=14)

        st.pyplot(fig)

    elif selected_chart == "Education Level":

        data = df["EdLevel"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(16, 12))
        wedges, texts, autotexts = ax1.pie(data,  autopct="%1.1f%%", shadow=False, startangle=90)
        ax1.axis("equal")
        for autotext in autotexts:
            autotext.set_fontsize(16)  # Adjust the font size as needed

        # Separate legend from the pie chart with a larger font size
        legend = ax1.legend(wedges, data.index, title="Education Levels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop={'size': 16})
        legend.set_title("Education Levels", prop={'size': 20})
        
        st.pyplot(fig1)



    # Assume df is your DataFrame containing the relevant data

    # Sidebar
    selected_category = st.sidebar.radio("Analyze mean salary based on: ", ["Experience", "Country", "Job", "Age", "Education Level"])

    # Main content based on the selected category
    st.write(f"#### Mean Salary Based On {selected_category}")

    if selected_category == "Experience":
        # Line chart: Experience vs Salary
        data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
        st.line_chart(data)

    elif selected_category == "Country":
        # Bar chart: Country vs Salary
        data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=False).reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('Country:N', sort=alt.EncodingSortField(field='Salary', op='mean', order='descending')),
            y='Salary:Q',
        )
        st.altair_chart(chart, use_container_width=True)

    elif selected_category == "Job":
        # Bar chart: Job vs Salary
        data = df.groupby(["DevType"])["Salary"].mean().sort_values(ascending=True).reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('DevType:N', sort=alt.EncodingSortField(field='Salary', op='mean', order='descending')),
            y='Salary:Q',
        )
        st.altair_chart(chart, use_container_width=True)

    elif selected_category == "Age":
        # Bar chart: Age vs Salary
        filtered_df = df[df["Age"] != 'Prefer not to say']
        data = filtered_df.groupby(["Age"])["Salary"].mean().sort_values(ascending=True).reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('Age:N', sort=alt.EncodingSortField(field='Salary', op='mean', order='descending')),
            y='Salary:Q',
        )
        st.altair_chart(chart, use_container_width=True)

    elif selected_category == "Education Level":
        # Bar chart: EdLevel vs Salary
        data = df.groupby(["EdLevel"])["Salary"].mean().sort_values(ascending=True).reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('EdLevel:N', sort=alt.EncodingSortField(field='Salary', op='mean', order='descending')),
            y='Salary:Q',
        )
        st.altair_chart(chart, use_container_width=True)



# Load the saved model and preprocessing parameters
lr_model = joblib.load('xgb_model.joblib')
features_columns = joblib.load('features_columns.joblib')
loaded_encoders = joblib.load('encoders.joblib')


# Streamlit app code
st.title("2023 Developer Salary Web App")
st.sidebar.title("2023 Developer Salary Web App")
st.sidebar.markdown("This application is based on data from the Stack Overflow Developer Survey 2023.")
st.write("Welcome! Please select the app mode in the sidebar to continue.")

# Radio button for user selection
app_mode = st.sidebar.radio("Select App Mode", ["Salary Analysis", "Salary Prediction"])




def user_input_features():
    Age= st.sidebar.selectbox("What is your age?",['Under 18 years old','18-24 years old', '25-34 years old','35-44 years old', '45-54 years old','55-64 years old',
       '65 years or older'],key="age")
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
    Database = st.sidebar.multiselect("Which database environments have you done extensive development work in over the past year",['BigQuery', 'Cassandra', 'Clickhouse', 'Cloud Firestore', 'Cockroachdb', 'Cosmos DB', 'Couch DB', 'Couchbase', 'Datomic', 'DuckDB', 'Dynamodb', 'Elasticsearch', 'Firebase Realtime Database', 'Firebird', 'H2', 'IBM DB2', 'InfluxDB', 'MariaDB', 'Microsoft Access', 'Microsoft SQL Server', 'MongoDB', 'MySQL', 'Neo4J', 'Oracle', 'PostgreSQL', 'RavenDB', 'Redis', 'Snowflake', 'Solr', 'SQLite', 'Supabase', 'TiDB', 'BigQuery', 'Cassandra', 'Clickhouse', 'Cloud Firestore', 'Cockroachdb', 'Cosmos DB', 'Couch DB', 'Couchbase', 'Datomic', 'DuckDB', 'Dynamodb', 'Elasticsearch', 'Firebase Realtime Database', 'Firebird', 'H2', 'IBM DB2', 'InfluxDB', 'MariaDB', 'Microsoft Access', 'Microsoft SQL Server', 'MongoDB', 'MySQL', 'Neo4J', 'Oracle', 'PostgreSQL', 'RavenDB', 'Redis', 'Snowflake', 'Solr', 'SQLite', 'Supabase', 'TiDB'],key="database")
    Platform = st.sidebar.multiselect("Which cloud platforms have you done extensive development work in over the past year?: ",['Amazon Web Services (AWS)', 'Cloudflare', 'Colocation', 'Digital Ocean', 'Firebase', 'Fly.io', 'Google Cloud', 'Heroku', 'Hetzner', 'IBM Cloud Or Watson', 'Linode', 'Managed Hosting', 'Microsoft Azure', 'Netlify', 'OpenShift', 'OpenStack', 'Oracle Cloud Infrastructure (OCI)', 'OVH', 'Render', 'Scaleway', 'Vercel', 'VMware', 'Vultr', 'Amazon Web Services (AWS)', 'Cloudflare', 'Colocation', 'Digital Ocean', 'Firebase', 'Fly.io', 'Google Cloud', 'Heroku', 'Hetzner', 'IBM Cloud Or Watson', 'Linode', 'Managed Hosting', 'Microsoft Azure', 'Netlify', 'OpenShift', 'OpenStack', 'Oracle Cloud Infrastructure (OCI)', 'OVH', 'Render', 'Scaleway', 'Vercel', 'VMware', 'Vultr'],key="platform")
    Webframe= st.sidebar.multiselect("Which web frameworks and web technologies have you done extensive development work in over the past year?: ",['Angular', 'AngularJS', 'ASP.NET', 'ASP.NET CORE', 'Blazor', 'CodeIgniter', 'Deno', 'Django', 'Drupal', 'Elm', 'Express', 'FastAPI', 'Fastify', 'Flask', 'Gatsby', 'jQuery', 'Laravel', 'Lit', 'NestJS', 'Next.js', 'Node.js', 'Nuxt.js', 'Phoenix', 'Play Framework ', 'Qwik', 'React', 'Remix', 'Ruby on Rails', 'Solid.js', 'Spring Boot', 'Svelte', 'Angular', 'AngularJS', 'ASP.NET', 'ASP.NET CORE', 'Blazor', 'CodeIgniter', 'Deno', 'Django', 'Drupal', 'Elm', 'Express', 'FastAPI', 'Fastify', 'Flask', 'Gatsby', 'jQuery', 'Laravel', 'Lit', 'NestJS', 'Next.js', 'Node.js', 'Nuxt.js', 'Phoenix', 'Play Framework ', 'Qwik', 'React', 'Remix', 'Ruby on Rails', 'Solid.js', 'Spring Boot', 'Svelte', 'Symfony', 'Vue.js', 'WordPress'],key="Webframe")
    LearnCode_str = ', '.join(LearnCode)
    Language_str = ', '.join(Language)
    Employment_str = ', '.join(Employment)
    Database_str =  ', '.join(Database)
    Platform_str =  ', '.join(Platform)
    Webframe_str=  ', '.join(Webframe)
    
    features={'Age': Age, 'RemoteWork': RemoteWork, 'EdLevel': EdLevel,  'YearsCodePro': YearsCodePro, 'Industry': Industry, 'DevType': DevType, 'OrgSize': OrgSize, 'Country': Country, 'Employment': Employment_str ,'LanguageHaveWorkedWith': Language_str, 'LearnCode': LearnCode_str,'DatabaseHaveWorkedWith':Database_str,'PlatformHaveWorkedWith':Platform_str,'WebframeHaveWorkedWith':Webframe_str}
    index = [0]
# Creating a DataFrame
    data_predict = pd.DataFrame(features, index=index)
    return data_predict
def pre_process(filtered_df):
    learn_code=filtered_df['LearnCode'].str.get_dummies(sep=';')
    languages = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
    Employ_situation = filtered_df['Employment'].str.get_dummies(sep=';')
    Database = filtered_df['DatabaseHaveWorkedWith'].str.get_dummies(sep=';')
    Platform = filtered_df['PlatformHaveWorkedWith'].str.get_dummies(sep=';')
    Webframe=filtered_df['WebframeHaveWorkedWith'].str.get_dummies(sep=';')
    Merge_data = (
    filtered_df
    .merge(Employ_situation, left_index=True, right_index=True)
    .merge(languages, left_index=True, right_index=True)
    .merge(learn_code, left_index=True, right_index=True)
    .merge(Database, left_index=True, right_index=True)
    .merge(Platform, left_index=True, right_index=True)
    .merge(Webframe, left_index=True, right_index=True)
)
    columns_to_drop = ["Employment", "LearnCode", "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith", 'WebframeHaveWorkedWith']
    New_data = Merge_data.drop(columns=columns_to_drop)
    table = New_data.copy()
    wanted_cols = table.select_dtypes(include=['object']).columns


    for col in wanted_cols:
        if col in loaded_encoders:
            table[col] = table[col].map(loaded_encoders[col])

        
   # Create an imputer object and specify the imputation strategy
    imputer = SimpleImputer(strategy="most_frequent")

   # Fit the imputer to the data
    imputer.fit(table)

   # Transform the data to impute missing values
    table_transformed = imputer.transform(table)

   # Convert the result back to a DataFrame
    table_transformed = pd.DataFrame(table_transformed, columns=table.columns)
    return table_transformed




if app_mode == "Salary Analysis":
    st.subheader("You've selected Salary Analysis.")
    st.sidebar.markdown("Please select the information you wish to know: ")
    df = load_data()
    show_explore()  
elif app_mode == "Salary Prediction":
    st.subheader("You've selected Salary Prediction.")
    # Add code for salary prediction here
    df = user_input_features()
    st.markdown("Please provide the required information in the sidebar and review your input details in the table below.")
    st.write(df)


    # Preprocess user input
    X_user = pre_process(df)

    # Ensure the input features have the same columns as the model was trained on
    missing_columns = set(features_columns) - set(X_user.columns)
    for col in missing_columns:
        X_user[col] = 0
    X_user = X_user.reindex(columns=features_columns, fill_value=0)

    # Make prediction
    y_pred = lr_model.predict(X_user)
    y_pred_original_scale = np.exp(y_pred)
    formatted_salary = "${:,.2f}".format(y_pred_original_scale[0])

    if st.button("Predict salary", key="predict"):
        st.markdown("Your yearly salary prediction is:")
        st.write(formatted_salary)



