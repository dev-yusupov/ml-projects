import re
import random

def split_sections_random(text):
    # Define regular expressions to find the starting and ending points of each section
    section_patterns = {
        "summary": re.compile(r"Professional\s*Summary", re.IGNORECASE),
        "education": re.compile(r"Education", re.IGNORECASE),
        "experience": re.compile(r"Professional\s*Experience", re.IGNORECASE),
        "skills": re.compile(r"Skills", re.IGNORECASE)
    }
    
    # Shuffle the order of sections
    section_order = list(section_patterns.keys())
    random.shuffle(section_order)
    
    # Initialize dictionary to store text for each section
    section_texts = {section: "" for section in section_order}
    
    # Find the starting and ending points of each section
    text_end = len(text)
    start_index = 0
    for section_name in section_order:
        section_start = section_patterns[section_name].search(text, start_index)
        if section_start:
            next_section_start = text_end
            for next_section_name, next_section_pattern in section_patterns.items():
                if next_section_name != section_name:
                    next_section_start = min(next_section_start, next_section_pattern.search(text, start_index) or text_end)
            section_texts[section_name] = text[section_start.end():next_section_start].strip()
            start_index = next_section_start
    
    return section_texts

def preprocess_text(text):
    # Split the text into sections in a random order
    text_sections = split_sections_random(text)
    
    # Initialize dictionary to store extracted information
    extracted_info = {
        "summary": "",
        "education": "",
        "experience": "",
        "skills": ""
    }
    
    # Assign text to appropriate sections
    for section_name, section_text in text_sections.items():
        if "professional summary" in section_text.lower():
            extracted_info["summary"] = section_text.strip()
        elif "education" in section_text.lower():
            extracted_info["education"] = section_text.strip()
        elif "experience" in section_text.lower():
            extracted_info["experience"] = section_text.strip()
        elif "skills" in section_text.lower():
            extracted_info["skills"] = section_text.strip()
    
    return extracted_info

text = """
BUSINESS DEVELOPMENT MANAGER/STAFFING MANAGER
Professional Summary
Innovative Manager seeks position offering opportunities for new professional and personal challenges. Self-starter with a positive, can-do attitude
who is driven to learn, improve and succeed. Over 10 years of proactive and indirect diverse recruiting and staffing experience.
Education and Training
2003
Bachelor of Science
 
: 
Health Science option in Health Management and Marketing
 
California State University of Hayward
 
ï¼​ 
City
 
, 
State
 
, 
United
States
Skill Highlights
Staffing management ability
Proven patience and self-discipline
Relationship and team building
Staff training and development
Critical thinking proficiency
Compensation/benefits administration
Skilled negotiator
Account management
Excellent written and verbal communicator
Enthusiastic team player
Problem solving
Billing
Attention to detail
Recruiting and selection techniques
Proficient communicator
Contract review
Cold calling
Hiring recommendations
Interviewing
Strategic planning
Multi-tasking ability
Skills
Â Proficient with Microsoft Word, Excel, PowerPoint, Access and Outlook Express. Â Optimizer, WFX, Stafferlink, Healthtrust, and Bullhorn.
Maintaining active databases of various hospital proprietary software technology systems.
Professional Experience
08/2006
 
to 
Current
Business Development Manager/Staffing Manager
 
Company Name
 
ï¼​ 
City
 
, 
State
 
Manage full cycle staffing, recruiting, and maintain a database
of clients and applicants that is aligned to the business opportunity in the market for recruitment, staffing, and placement . Responsible for job
postings, hiring, interviewing, and training new employees.Â 
 Generate new accounts by implementing effective networking and content marketing
strategies. 
Manage budget forecasting, goal setting and performance reporting for all accounts. 
Negotiate rates to cut costs and benefit corporate
partnerships
 
.Â 
 
Demonstrate knowledge of HIPAA Privacy and Security Regulations. 
Conduct reference and background checks on all job
applicants. 
Developed creative recruiting strategies that met anticipated staffing needs. 
Communicate the duties, compensation, benefits and
working conditions to all potential candidates.
 
Contact all job applicants to inform them of their application status. Work with Director of Nursing
and Human Resource Directors to ensure all healthcare organization is able to support business growth. Coach and counsel employees regarding
attendance & performance; mediate employee disputes and complaints.Â 
 Respond 
Onboard new employees in the time reporting and payroll
systems
 
. 
Manage payroll and time and attendance systems.
05/2003
 
to 
08/2008
Staffing Coordinator
 
Company Name
 
ï¼​ 
City
 
, 
State
 
Created and maintained all absentee calendars, agency nurse schedules and staff meeting
minutes. 
Maintained all confidential personnel files, licensing and CPR compliance records. Develop computerized schedules for assigned nursing
units based on established staffing patterns, policies, approved employee preferences, and managers' requests. Revises and adjusts unit schedules
as needed in consultation with nurse managers.Â Proactively adjusts and allocates core, registry, and float nursing personnel to provide adequate
coverage to clinics and inpatient areas to strategically meet real-time staffing requirements in the most cost-effective manner
05/2003
 
to 
08/2008
Staffing Manager
 
Company Name
 
ï¼​ 
City
 
, 
State
 
Manage full cycle staffing, recruiting, and maintain a database of clients and applicants that is
aligned to the business opportunity in the market for recruitment, staffing, and placement . Responsible for job postings, hiring, interviewing, and
training new employees.Â Generate new accounts by implementing effective networking and content marketing strategies.Manage budget
forecasting, goal setting and performance reporting for all accounts.Â Negotiate rates to cut costs and benefit corporate
partnerships.Â Demonstrate knowledge of HIPAA Privacy and Security Regulations.Â Conduct reference and background checks on all job
applicants.Â Developed creative recruiting strategies that met anticipated staffing needs.Â Communicate the duties, compensation, benefits and
working conditions to all potential candidates.Â Contact all job applicants to inform them of their application status. Work with Director of Nursing
and Human Resource Directors to ensure all healthcare organization is able to support business growth. Coach and counsel employees regarding
attendance & performance; mediate employee disputes and complaints.Â RespondÂ Onboard new employees in the time reporting and payroll
systemsÂ .Â Manage payroll and time and attendance systems.
"""

processed_text = preprocess_text(text)
print(processed_text)
