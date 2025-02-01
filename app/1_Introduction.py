import streamlit as st

st.set_page_config(page_title = "Introduction", layout = "wide")

st.title(":syringe::microbe: Explainer Dashboard for Predictions of H1N1 and Seasonal Flu Vaccination Uptake")

intro1, intro2 = st.columns([0.25, 0.75], vertical_alignment = "center")

intro1.image("images/vaccine.jpeg")
intro2.markdown(
    '''
    ### **Overview**

    [Immunisation](https://www.who.int/health-topics/vaccines-and-immunization#tab=tab_1) :link: is a fundamental aspect of primary health care and an indisputable human right. 
    Vaccines are critical to the prevention and control of infectious disease outbreaks, and underpin global health security.

    Vaccines work by training your immune system to create antibodies and assist your body’s natural defences to help build protection.
    Because vaccines contain only killed or weakened forms of germs like viruses or bacteria, they do not cause the disease or put you at risk of its complications.

    There are vaccines for more than 20 [life-threatening diseases](https://www.who.int/teams/immunization-vaccines-and-biologicals/diseases) :link: including: cholera, typhoid, influenza, rabies, measles, mumps and rubella (MMR). 
    Immunisation against these diseases prevents 3-3.5 million deaths each year.

    However, vaccines have always been a polarising topic. [Anti-vaccinationism](https://en.wikipedia.org/wiki/Vaccine_hesitancy) :link: (commonly known today as "anti-vax" or "anti-vaxxers"), refers to the complete opposition of vaccines - 
    often propogated through conspiracy theories, mis/dis information and fringe science. Such vaccine hesitancy has led to increasingly large numbers of the population to 
    delay, or outright refuse vaccinations, which as a result prevents heard immunity and leads to increased outbreaks and death from these diseases. 

    Vaccine hesitancy is characterised by the [World Health Organisation](https://www.who.int/news-room/spotlight/ten-threats-to-global-health-in-2019) :link: as one of the top-10 global health threats.
    ''')

st.markdown('''---''')

virus1, virus2 = st.columns([0.75, 0.25], vertical_alignment = "top")

virus1.markdown(
    f'''
    ### **Task Outline**
    
    In the beginning of spring 2009, a major respiratory disease pandemic caused by the [H1N1 influenza virus](https://en.wikipedia.org/wiki/Influenza_A_virus_subtype_H1N1) :link:, colloquially named "swine flu," swept across the world. 
    Researchers estimate that in the first year, it was responsible for between [151,000 to 575,000 deaths globally](https://zenodo.org/records/1260250) :link:. 
    A vaccine for the H1N1 flu virus quickly followed and became publicly available in October 2009. 
    
    In late 2009 and early 2010, the United States conducted the [National 2009 H1N1 Flu Survey](https://www.cdc.gov/nchs/nis/data_files_h1n1.htm) :link:. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. 
    These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. 

    Using these features, a model has been developed to predict how likely an individual is to get vaccinated against the H1N1 influenza virus and seasonal flu.
    
    ---

    ### **Objective**

    This explainer dashboard aims to provide guidance for future public health efforts and increase the understanding of how these 
    protected, behavioural and opinion-based characteristics are associated with personal vaccination patterns.

    Using this dashboard, you will be able to determine:

    <a href="http://localhost:8501/Modelling" target="_self">Modelling</a>
    - How accurately the model can predict if an individual receives their H1N1 and seasonal flu vaccines
        
    - Which features are most important in the model's predictions, and how do the predictions change as these feature values change

    - How fair and unbiased the model is
    
    <a href="http://localhost:8501/Data" target="_self">Data</a>

    - Is the composition of the data suitable for modelling purposes
    ''', unsafe_allow_html = True)
virus2.image("images/h1n1.jpeg")