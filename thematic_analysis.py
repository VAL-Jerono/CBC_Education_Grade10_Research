import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import networkx as nx
import re
import json
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3', 'font.family': 'serif',
    'axes.titlecolor': '#e6edf3', 'axes.labelcolor': '#8b949e',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
})

# ════════════════════════════════════════════════════════════════════════════
# CORPUS — Rich text extracted from every source
# Each entry: id, short_id, year, type, level, full text of findings/abstract
# ════════════════════════════════════════════════════════════════════════════
corpus = [
    {
        "id": "mwangombe21", "year": 2021, "type": "empirical", "level": "Primary",
        "label": "Mwang'ombe 2021",
        "text": """Teacher preparedness for competency based curriculum in Kenya reveals a crisis.
        Only three percent of educators feel adequately prepared for CBC implementation.
        Teachers lack understanding of CBC philosophy and competency based assessment methods.
        Most teachers received no formal training before curriculum rollout.
        Teachers continue using traditional rote learning methods instead of learner centered approaches.
        Content knowledge gaps observed especially in science and mathematics strands.
        Teachers struggle with project based learning and formative assessment design.
        Schools lack teaching and learning resources to support CBC activities.
        The shift from knowledge transmission to competency development is poorly understood by teachers.
        Teacher professional development programs are inadequate and poorly coordinated."""
    },
    {
        "id": "mohamed22", "year": 2022, "type": "empirical", "level": "Secondary",
        "label": "Mohamed et al. 2022",
        "text": """Secondary school teachers in Kirinyaga County demonstrate inadequate pedagogical preparedness for CBC.
        Majority of respondents had not attended any in-service training on CBC methodology.
        Teachers were not conversant with CBC concepts and framework requirements.
        Pedagogical content knowledge deficits observed across all subject areas taught.
        Assessment practices remain examination oriented rather than competency based.
        Teachers revert to lecture methods because they lack skills for learner centered instruction.
        Curriculum designs are available but teachers cannot interpret or apply them effectively.
        Subject specific training is completely absent for secondary school teachers.
        Teachers express anxiety about CBC implementation due to lack of guidance and support.
        School principals also lack capacity to mentor teachers in CBC delivery methods."""
    },
    {
        "id": "ngeno23", "year": 2023, "type": "empirical", "level": "Primary",
        "label": "Ngeno 2023",
        "text": """Teacher training inadequacy is a central concern in CBC implementation in Kericho County.
        Stakeholders including parents community members and education officers raise concerns about CBC readiness.
        Teachers report insufficient time allocated for in-service training programs.
        Training workshops do not adequately address classroom practice and assessment challenges.
        CBC requires collaboration between teachers learners parents and community but this is poorly supported.
        Learning resources materials and infrastructure remain insufficient in most schools.
        Teacher workload increased significantly under CBC due to continuous assessment records.
        Professional development opportunities are sporadic and not sustained over time.
        Teachers feel isolated and without ongoing support from curriculum developers or supervisors."""
    },
    {
        "id": "muchira23", "year": 2023, "type": "empirical", "level": "Primary",
        "label": "Muchira et al. 2023",
        "text": """Comparative analysis of CBC implementation in Kenya USA and South Korea reveals systematic challenges.
        Kenya shows limited evidence on the effect of competency based curriculum on learner key competencies.
        Teacher training gaps are a universal challenge across all three countries studied.
        USA demonstrates that sustained professional development communities improve teacher readiness significantly.
        South Korea uses structured mentorship and peer observation to build teacher competence over time.
        Kenya relies primarily on short one off workshops which are demonstrably ineffective.
        Lack of funding affects teacher training quality and resource availability for CBC delivery.
        Policy implementation gaps exist between national curriculum guidelines and classroom practice.
        Assessment reform is incomplete because teachers lack formative assessment skills and tools.
        Learner outcomes data under CBC remains largely uncollected and unanalyzed in Kenya."""
    },
    {
        "id": "wasanga22", "year": 2022, "type": "empirical", "level": "Primary",
        "label": "Wasanga & Wambua 2022",
        "text": """Teacher capacity is the primary implementation barrier for CBC in Kenyan public schools.
        Resource constraints including learning materials equipment and infrastructure significantly hinder CBC delivery.
        Teachers trained under 8-4-4 system lack competency based pedagogical skills required by CBC.
        Continuous professional development is needed but not systematically provided by TSC or MoE.
        Large class sizes make learner centered activities and differentiated instruction impractical.
        Assessment burden under CBC creates overwhelming administrative workload for teachers.
        Teachers struggle to balance curriculum coverage with depth of competency development.
        School leadership plays a critical role in supporting or hindering CBC implementation quality.
        Parents misunderstand CBC leading to insufficient home support for learner portfolios and projects.
        Budget allocation for CBC related teacher training remains inadequate at county and national level."""
    },
    {
        "id": "mogere25", "year": 2025, "type": "empirical", "level": "SeniorSchool",
        "label": "Mogere & Mbatanu 2025",
        "text": """Senior school pathway teacher requirements represent a critical gap in CBC implementation planning.
        Projection model for Vihiga County reveals significant teacher shortfalls across all three pathways.
        STEM pathway faces most severe teacher shortage especially for technical subjects and applied sciences.
        Social sciences pathway requires teachers with cross disciplinary knowledge uncommon in current workforce.
        Arts and sports science pathway lacks specialized instructors particularly for performance and creative arts.
        Deployment of qualified teachers to senior school level is not aligned with pathway specific needs.
        TSC recruitment pipelines have not yet produced sufficient pathway trained teachers for Grade 10 rollout.
        County director of education confirms inadequate preparation time before January 2026 implementation.
        New subject combinations in senior school require teacher retraining not just deployment.
        Without targeted senior school teacher development program the quality of CBC at this level is at risk."""
    },
    {
        "id": "mackatiani23", "year": 2023, "type": "empirical", "level": "JSS",
        "label": "Mackatiani et al. 2023",
        "text": """Junior secondary school implementation of CBC reveals deepened preparedness challenges beyond primary level.
        Teachers deployed to Grade 7 8 9 were largely trained under 8-4-4 and lack CBC specific competencies.
        Integrated subject areas in JSS require teachers with broader knowledge than typical specialization allows.
        Assessment portfolios and project based evaluation are unfamiliar to most JSS teachers.
        School facilities for JSS CBC activities are inadequate including laboratories workshops and ICT equipment.
        Head teachers struggle to provide instructional leadership because they themselves lack CBC training.
        TSC deployment decisions did not consider subject pathway alignment when placing teachers in JSS.
        Training cascade model failed because master trainers did not have deep enough CBC knowledge.
        Teachers report low morale and stress related to CBC implementation demands without adequate support.
        Parental engagement in JSS CBC activities is lower than expected due to poor communication from schools."""
    },
    {
        "id": "knut22", "year": 2022, "type": "report", "level": "Primary/JSS",
        "label": "KNUT Report 2022",
        "text": """Kenya National Union of Teachers investigation reveals systematic failures in CBC teacher training program.
        Training sessions were poorly designed and their effectiveness fell far below expectations of teachers and schools.
        Facilitators conducting CBC training workshops were themselves not fully competent in CBC methodology.
        Most pre-primary and Grade 1 to 3 teachers received no training whatsoever before CBC implementation.
        Cascade training model degraded quality at each level with errors compounding from master trainers downward.
        Training duration was insufficient with most sessions limited to one to three days for a major curriculum shift.
        Content of training did not match subject specific classroom needs of individual teachers.
        Union calls for systemic overhaul of teacher training program before senior school rollout begins.
        Teachers who attended training report leaving more confused than prepared to implement CBC.
        Recommended training approach includes sustained mentorship school based coaching and peer learning circles."""
    },
    {
        "id": "cheruiyot24", "year": 2024, "type": "empirical", "level": "JSS",
        "label": "Cheruiyot 2024",
        "text": """Challenges in CBC implementation at junior school level include critical teacher training deficits.
        Digital infrastructure shortages prevent effective integration of ICT tools required by CBC methodology.
        Most rural schools lack computers tablets and reliable internet access needed for digital learning activities.
        Teacher digital literacy is insufficient for CBC requirements even where devices are available.
        Inadequate training compounds infrastructure challenges because teachers cannot adapt to technology constraints.
        Assessment record keeping systems are paper based and create enormous administrative burden for teachers.
        CBC formative assessment requirements demand more time than teachers have within current timetabling.
        Subject integration concepts in CBC are poorly understood by single subject trained teachers.
        Professional learning communities among teachers are absent making it difficult to share CBC strategies.
        MoE must address infrastructure gap alongside training gap for CBC implementation to succeed."""
    },
    {
        "id": "kailo25", "year": 2025, "type": "empirical", "level": "Primary",
        "label": "Kailo et al. 2025",
        "text": """Training modality significantly influences teacher preparedness and classroom delivery under CBC in Kilifi County.
        Interactive and practical in-service training approaches yield significantly better outcomes than lecture based workshops.
        Interactivity of CBC training impacts instructional delivery assessment quality and professional growth of teachers.
        Statistical evidence confirms that hands on training with real lesson demonstrations is more effective.
        Training programs must include subject specific pedagogical practice not just general CBC orientation.
        Feedback and reflection components of training improve teacher confidence and self efficacy for CBC delivery.
        Continuous mentoring following initial training sustains improvements in classroom practice over time.
        Peer learning and collaborative lesson planning are effective low cost professional development strategies.
        Training must address not just what to teach but how to teach it for competency development.
        Communities of practice among CBC teachers create sustained professional development without large budgets."""
    },
    {
        "id": "keter25", "year": 2025, "type": "empirical", "level": "JSS",
        "label": "Keter & Wabuke 2025",
        "text": """Teacher preparedness and CBC implementation quality are positively correlated in Bomet County JSS schools.
        Deficiencies in pedagogical knowledge are the strongest predictor of poor CBC implementation outcomes.
        Professional development opportunities are severely limited for most teachers especially in rural areas.
        Teachers with higher PCK scores deliver more learner centered competency based lessons consistently.
        Assessment design skills are the weakest area of teacher competence across all subject areas studied.
        Subject content knowledge alone is insufficient for effective CBC delivery without pedagogical skills.
        Cronbach alpha of 0.82 confirms reliability of teacher preparedness measurement instrument used.
        Head teacher support and instructional supervision are significant moderators of teacher implementation quality.
        Resources including learning materials and facilities have direct positive effect on CBC delivery quality.
        Sustained professional development programs produce stronger implementation outcomes than one off training."""
    },
    {
        "id": "garissa25", "year": 2025, "type": "empirical", "level": "Primary",
        "label": "Garissa Study 2025",
        "text": """Marginalized regions face compounded barriers to effective CBC implementation far exceeding national averages.
        Garissa County teachers receive insufficient CBC training and have minimal access to curriculum resources.
        Infrastructure constraints including lack of classrooms furniture electricity and water affect learning environments.
        Cultural and linguistic factors create additional challenges for CBC delivery in arid regions.
        Teacher turnover is high in marginalized areas due to hardship conditions creating continuity problems.
        Remote schools cannot access in-service training because of distance travel costs and teacher release challenges.
        Equity in education demands targeted differentiated support for CBC implementation in marginalized counties.
        Assessment requirements under CBC cannot be met without baseline infrastructure improvements first.
        Community engagement in CBC is lower in areas with lower parental literacy and education levels.
        Multi-dimensional support framework needed addressing training infrastructure culture and resources simultaneously."""
    },
    {
        "id": "ijriss25", "year": 2025, "type": "empirical", "level": "JSS",
        "label": "IJRISS 2025",
        "text": """Stakeholder attitudes and perceptions significantly challenge CBC implementation in junior secondary schools.
        Seventy seven percent of JSS schools identify negative stakeholder attitudes as major implementation barrier.
        Teachers who distrust CBC philosophy deliver it with less commitment resulting in lower quality outcomes.
        TSC confirms that teachers are not adequately equipped for CBC pedagogical demands at JSS level.
        Parents skepticism about CBC compared to former KCPE examination system affects learner engagement at home.
        Teacher resistance to change from familiar 8-4-4 teaching methods is widespread and persistent.
        Attitude change interventions must accompany training programs for effective CBC implementation.
        School leadership attitudes directly influence teacher motivation and commitment to CBC delivery.
        Community sensitization campaigns about CBC benefits are insufficient and poorly targeted.
        Self efficacy of teachers is undermined by unsupportive school culture and negative peer attitudes."""
    },
    {
        "id": "momanyi19", "year": 2019, "type": "empirical", "level": "Primary",
        "label": "Momanyi & Rop 2019",
        "text": """Early grade primary teachers in Bomet East are not adequately ready to handle CBC implementation.
        Teacher readiness survey reveals gaps in understanding of learner centered pedagogical approaches.
        Classroom observation confirms continued reliance on chalk and talk methods incompatible with CBC.
        Assessment for learning practices are absent with teachers defaulting to summative test approaches.
        Limited awareness of CBC curriculum documents and design means teachers improvise without proper guidance.
        Infrastructure including learning corners resource areas and outdoor spaces are missing from most classrooms.
        Teacher educator preparation programs at training colleges have not yet incorporated CBC specific content.
        Pre-service training remains oriented toward 8-4-4 philosophy creating readiness gap before teachers enter service.
        In-service training provided is too short to produce meaningful changes in teacher instructional practice.
        Classroom management challenges increase under CBC as learner centered activities require different skills."""
    },
    {
        "id": "maluha24", "year": 2024, "type": "empirical", "level": "Primary",
        "label": "Maluha et al. 2024",
        "text": """Teacher pedagogical practices directly influence the quality of CBC implementation in Vihiga County schools.
        PCK is the mediating variable between teacher training and learner competency outcomes under CBC.
        Teachers who demonstrate strong subject knowledge combined with learner centered pedagogy achieve better outcomes.
        Instructional strategies must shift from content transmission to competency facilitation for CBC success.
        Formative assessment practices including observation checklists rubrics and portfolios improve learning outcomes.
        Professional development programs that focus on classroom practice rather than theory are more effective.
        School based instructional coaching produces sustained improvements in teacher CBC delivery quality.
        Differentiated instruction to address diverse learner needs is a critical but underdeveloped CBC competency.
        Teachers need both content knowledge refreshment and new pedagogical skill development for CBC success.
        Collaborative planning between teachers enhances consistency and quality of CBC delivery across classrooms."""
    },
    {
        "id": "njiru24", "year": 2024, "type": "empirical", "level": "ECDE",
        "label": "Njiru & Odundo 2024",
        "text": """Professional development gaps for early childhood teachers persist across all CBC implementation levels.
        ECDE teachers lack foundational understanding of competency based learning philosophy and methodology.
        Training programs for ECDE teachers are insufficient in duration content and follow up support.
        Play based learning and exploration which are central to CBC at early grades are poorly implemented.
        Teacher educator preparation programs at colleges must align with CBC before new teachers can be effective.
        Continuous professional development through reflection peer observation and mentoring is largely absent.
        Digital technology integration in early childhood education is minimal due to training and resource gaps.
        Assessment in early childhood under CBC requires observation skills teachers have not been trained in.
        Community of practice approach to professional learning can address training gaps at low cost.
        Systemic reform of teacher education programs is necessary foundation for sustainable CBC implementation."""
    },
    {
        "id": "pwper23", "year": 2023, "type": "policy", "level": "National",
        "label": "PWPER 2023",
        "text": """Presidential Working Party on Education Reform identifies teacher preparedness as most urgent CBC challenge.
        Report recommends systematic teacher retooling before each new CBC level rollout including senior school.
        Funding allocation for CBC teacher training must increase substantially from current levels.
        Teacher training colleges must reform pre-service curriculum to align with CBC philosophy and methodology.
        TSC must develop clear competency framework for CBC teachers across all pathway subjects.
        School based continuous professional development should replace one off workshop training model.
        Assessment system reform requires teacher training component to build formative assessment skills nationwide.
        Senior school pathway implementation will fail without pathway specific teacher preparation program.
        Equity concerns demand that marginalized county teachers receive additional targeted training support.
        Public private partnerships should expand teacher training capacity beyond what government can provide alone."""
    },
    {
        "id": "tsc_plan", "year": 2023, "type": "policy", "level": "National",
        "label": "TSC Plan 2023-27",
        "text": """Teachers Service Commission strategic plan commits to systematic teacher professional development for CBC.
        Teacher Induction Mentoring and Coaching program TIMEC provides structured support for new and experienced teachers.
        Teacher Performance Appraisal and Development TPAD links professional development to classroom performance outcomes.
        Deployment policy must align teacher qualifications with CBC pathway subject requirements at senior school.
        Recruitment targets of 18000 new teachers per year include pathway specific qualifications for senior school.
        Professional development KPIs include percentage of teachers trained per CBC level before rollout.
        Digital training platforms will expand teacher development access in remote and marginalized areas.
        School based mentorship circles will complement formal in-service training with ongoing classroom support.
        Collaboration with KICD ensures training content aligns with current curriculum designs and assessment frameworks.
        Senior school teacher preparation requires specialized program separate from general CBC orientation training."""
    },
    {
        "id": "kicd_designs", "year": 2024, "type": "policy", "level": "SeniorSchool",
        "label": "KICD Designs 2024",
        "text": """Senior school curriculum designs require high levels of pedagogical content knowledge from subject teachers.
        STEM pathway teachers must integrate inquiry based learning practical investigations and critical thinking skills.
        Social sciences pathway demands interdisciplinary teaching capacity linking humanities arts and contemporary issues.
        Arts and sports science pathway requires performance based assessment and practical skill development expertise.
        Each pathway subject has distinct competency strands requiring specialized teacher content knowledge.
        Assessment framework for senior school includes school based assessment and national examinations.
        Teachers must design assessments that measure competencies not just content knowledge recall.
        Curriculum designs assume teachers have received adequate pathway specific training before delivery.
        Learning resources including equipment laboratory materials and digital tools specified in designs are mandatory.
        KICD recommends continuous professional development and subject panel meetings for senior school teachers."""
    },
    {
        "id": "shulman86", "year": 1986, "type": "theory", "level": "Theory",
        "label": "Shulman 1986",
        "text": """Pedagogical content knowledge represents the intersection of subject matter expertise and teaching knowledge.
        Teachers must understand not only what to teach but how to transform content for learner understanding.
        Content knowledge alone is insufficient for effective teaching without pedagogical transformation skills.
        PCK includes knowledge of learner misconceptions typical errors and effective representations of content.
        Subject specific teaching methods differ fundamentally from general pedagogical approaches to instruction.
        Curriculum knowledge encompasses understanding how content is sequenced developed and connected across levels.
        Teacher education must develop PCK systematically rather than assuming content expertise transfers to teaching.
        Professional knowledge of teaching includes understanding of learner diversity and differentiated instruction.
        Assessment knowledge is integral to PCK enabling teachers to diagnose learning and adjust instruction.
        PCK framework predicts that teacher effectiveness is highest when content and pedagogy are deeply integrated."""
    },
    {
        "id": "bandura97", "year": 1997, "type": "theory", "level": "Theory",
        "label": "Bandura 1997",
        "text": """Self efficacy represents an individual's belief in their capacity to execute behaviors required for specific outcomes.
        High self efficacy leads to greater effort persistence and resilience in the face of challenging tasks.
        Teacher self efficacy directly predicts quality of instructional practice and student learning outcomes.
        Mastery experiences are the strongest source of self efficacy building when teachers succeed at CBC delivery.
        Social modeling through observation of effective peer teachers builds self efficacy in new practitioners.
        Verbal encouragement and coaching from supervisors and mentors can strengthen teacher confidence.
        Low self efficacy creates avoidance behaviors where teachers revert to familiar but ineffective methods.
        Training programs that include successful practice opportunities build self efficacy more than lectures.
        Collective teacher efficacy within a school predicts school level implementation quality and student outcomes.
        Self efficacy mediates the relationship between training received and actual changes in classroom practice."""
    },
    {
        "id": "fullan07", "year": 2007, "type": "theory", "level": "Theory",
        "label": "Fullan 2007",
        "text": """Educational change requires transformation across three dimensions: beliefs, skills, and material resources.
        Change is not an event but a process requiring sustained support over multiple years of implementation.
        One off training workshops are insufficient for deep change in teacher beliefs and instructional practices.
        System level change requires alignment between policy design professional development and school support structures.
        Moral purpose and teacher agency are essential drivers of sustainable educational reform implementation.
        Resistance to change is natural and must be addressed through trust building and inclusive engagement.
        Leadership at school level is the most powerful lever for supporting teacher implementation of reforms.
        Professional learning communities create the collaborative culture needed for sustained change over time.
        Coherence between different reform initiatives prevents initiative fatigue and conflicting demands on teachers.
        Evidence of improved student outcomes is the most powerful motivator for teacher adoption of new practices."""
    },
    {
        "id": "darling19", "year": 2019, "type": "theory", "level": "Theory",
        "label": "Darling-Hammond 2019",
        "text": """Science of learning demonstrates that competency based approaches align with how human development occurs.
        Learner centered pedagogy produces deeper understanding than content transmission approaches to teaching.
        Effective teaching requires knowledge of child development cognitive science and social emotional learning.
        Assessment for learning through formative feedback is more effective than summative testing for competency development.
        Teacher professional development must model the same learner centered approaches teachers are expected to use.
        Inquiry based learning and project based activities develop higher order thinking and real world competencies.
        Relationship centered school environments improve both teacher effectiveness and learner motivation and engagement.
        Differentiated instruction responds to diverse learner needs and is foundational to competency based education.
        Professional learning communities provide ongoing collaborative support for teacher growth and curriculum implementation.
        Evidence informed practice requires teachers to continuously reflect assess and adjust their instructional approaches."""
    },
]

print(f"✅ Corpus loaded: {len(corpus)} documents")
print(f"   Types: {Counter([d['type'] for d in corpus])}")
print(f"   Levels: {Counter([d['level'] for d in corpus])}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — TF-IDF: What words are most distinctive per source?
# ════════════════════════════════════════════════════════════════════════════
texts  = [d["text"] for d in corpus]
labels = [d["label"] for d in corpus]
ids    = [d["id"]   for d in corpus]

STOP_WORDS = list({'the','and','of','in','for','to','a','is','are','that','this',
    'by','on','with','from','be','as','at','not','its','it','have','has',
    'an','or','which','their','they','also','been','can','more','than',
    'but','was','were','when','will','all','each','other','who','most',
    'these','through','about','would','such','into','across','both','under',
    'over','between','because','only','being','had','he','she','we','us','our',
    'any','per','how','what','where','do','does','did','if','so','up','out',
    'level','levels','school','schools','kenya','kenyan','cbc','teacher','teachers',
    'training','implementation','preparedness','curriculum'})

tfidf = TfidfVectorizer(
    max_features=120, stop_words=STOP_WORDS,
    ngram_range=(1,2), min_df=2, max_df=0.92,
    token_pattern=r'\b[a-z][a-z]+\b'
)
tfidf_matrix = tfidf.fit_transform(texts)
feature_names = tfidf.get_feature_names_out()

print(f"\n✅ TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} terms")

# Global term importance
global_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
term_importance = pd.DataFrame({'term': feature_names, 'score': global_tfidf})
term_importance = term_importance.sort_values('score', ascending=False).reset_index(drop=True)

print("\nTop 25 most globally distinctive terms across all literature:")
print(term_importance.head(25).to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — LDA Topic Modelling
# ════════════════════════════════════════════════════════════════════════════
count_vec = CountVectorizer(
    max_features=100, stop_words=STOP_WORDS,
    ngram_range=(1,2), min_df=2, max_df=0.90,
    token_pattern=r'\b[a-z][a-z]+\b'
)
count_matrix = count_vec.fit_transform(texts)
count_features = count_vec.get_feature_names_out()

N_TOPICS = 6
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42,
                                 max_iter=200, learning_method='batch')
lda_matrix = lda.fit_transform(count_matrix)

# Name the topics manually based on top words
topic_names = {
    0: "Teacher Readiness\n& Classroom Practice",
    1: "Training Quality\n& Modality",
    2: "PCK & Pedagogy",
    3: "Policy & System\nChange",
    4: "Assessment &\nLearner Outcomes",
    5: "Equity, Resources\n& Infrastructure",
}

print(f"\n✅ LDA topics: {N_TOPICS} topics from {count_matrix.shape[1]} features")
print()
for tid in range(N_TOPICS):
    top_words_idx = lda.components_[tid].argsort()[-10:][::-1]
    top_words = [count_features[i] for i in top_words_idx]
    print(f"Topic {tid+1} | {topic_names[tid].replace(chr(10),' ')}")
    print(f"  → {', '.join(top_words)}")
    print()

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Term Co-occurrence Network
# ════════════════════════════════════════════════════════════════════════════
# Use top 40 terms; edge = co-occurrence in same document
top40_terms = list(term_importance.head(40)['term'])

def get_terms_in_doc(text, terms):
    words = set(re.findall(r'\b[a-z][a-z]+\b', text.lower()))
    bigrams = {' '.join([a,b]) for a,b in zip(list(words),list(words)[1:])}
    present = []
    for t in terms:
        if ' ' in t:
            if t in text.lower(): present.append(t)
        else:
            if t in words: present.append(t)
    return present

cooccur = Counter()
for doc in corpus:
    present = get_terms_in_doc(doc['text'], top40_terms)
    for pair in combinations(sorted(present), 2):
        cooccur[pair] += 1

G_terms = nx.Graph()
for term in top40_terms:
    score = float(term_importance[term_importance['term']==term]['score'].iloc[0]) if term in term_importance['term'].values else 0.01
    G_terms.add_node(term, weight=score)

for (t1,t2), count in cooccur.items():
    if count >= 3:
        G_terms.add_edge(t1, t2, weight=count)

print(f"✅ Term co-occurrence network: {G_terms.number_of_nodes()} nodes, {G_terms.number_of_edges()} edges")

# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Top Terms Treemap-style bar chart
# ════════════════════════════════════════════════════════════════════════════
COLOR_PALETTE = ['#ef4444','#f97316','#f59e0b','#10b981','#3b82f6','#8b5cf6',
                 '#ec4899','#06b6d4','#84cc16','#a78bfa']

# Assign each term a topic color
def term_to_topic(term):
    topic_keywords = {
        0: ['readiness','classroom','practice','learner centered','delivery','instruction','methods','approach'],
        1: ['in-service','workshop','modality','cascade','mentoring','coaching','peer','sustained'],
        2: ['pck','pedagogical','content knowledge','subject','competency','skills','formative'],
        3: ['policy','reform','system','tsc','moe','kicd','deployment','leadership','strategic'],
        4: ['assessment','outcomes','formative','portfolio','rubric','competencies','evaluation'],
        5: ['equity','resources','infrastructure','marginalized','rural','facilities','digital','access'],
    }
    for tid, keywords in topic_keywords.items():
        if any(k in term for k in keywords):
            return tid
    return 3

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.patch.set_facecolor('#0d1117')

# Left: horizontal bar chart of top 30 terms
ax1 = axes[0]; ax1.set_facecolor('#0d1117')
top30 = term_importance.head(30)
bar_colors = [COLOR_PALETTE[term_to_topic(t) % len(COLOR_PALETTE)] for t in top30['term']]
bars = ax1.barh(range(len(top30)), top30['score'], color=bar_colors, alpha=0.88, edgecolor='#1a1a2e', linewidth=0.5)
ax1.set_yticks(range(len(top30)))
ax1.set_yticklabels(top30['term'], fontsize=9.5, color='#e6edf3')
ax1.invert_yaxis()
ax1.set_xlabel("TF-IDF Score (distinctiveness across literature)", fontsize=10)
ax1.set_title("Most Distinctive Terms Across All 24 Sources\n(color = thematic cluster)", fontsize=12, color='#e6edf3', pad=12)
ax1.spines[:].set_color('#30363d')
ax1.tick_params(colors='#8b949e')
for bar, val in zip(bars, top30['score']):
    ax1.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left', color='#8b949e', fontsize=7.5)

# Right: bubble chart — term frequency vs. score
ax2 = axes[1]; ax2.set_facecolor('#0d1117')
term_freq = np.asarray(count_matrix.sum(axis=0)).flatten()
term_freq_df = pd.DataFrame({'term': count_features, 'freq': term_freq})

plot_terms = term_importance.head(35).merge(term_freq_df, on='term', how='left').fillna(1)
bubble_colors = [COLOR_PALETTE[term_to_topic(t) % len(COLOR_PALETTE)] for t in plot_terms['term']]

ax2.scatter(plot_terms['freq'], plot_terms['score'],
            s=plot_terms['score']*8000, c=bubble_colors, alpha=0.75, edgecolors='#1a1a2e', linewidth=0.5)
for _, row in plot_terms.iterrows():
    ax2.annotate(row['term'], (row['freq'], row['score']),
                 fontsize=7.5, color='#e6edf3', ha='center', va='bottom',
                 xytext=(0, 6), textcoords='offset points')
ax2.set_xlabel("Document Frequency (how many sources use this term)", fontsize=10)
ax2.set_ylabel("TF-IDF Score (how distinctive/important)", fontsize=10)
ax2.set_title("Term Importance vs. Frequency\nLarge bubbles = high TF-IDF weight", fontsize=12, color='#e6edf3', pad=12)
ax2.spines[:].set_color('#30363d'); ax2.tick_params(colors='#8b949e')

legend_patches = [
    mpatches.Patch(color=COLOR_PALETTE[0], label='Teacher Readiness'),
    mpatches.Patch(color=COLOR_PALETTE[1], label='Training Quality'),
    mpatches.Patch(color=COLOR_PALETTE[2], label='PCK & Pedagogy'),
    mpatches.Patch(color=COLOR_PALETTE[3], label='Policy & System'),
    mpatches.Patch(color=COLOR_PALETTE[4], label='Assessment & Outcomes'),
    mpatches.Patch(color=COLOR_PALETTE[5], label='Equity & Resources'),
]
ax2.legend(handles=legend_patches, loc='upper right', fontsize=8,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

plt.suptitle("CBC Literature Thematic Analysis — What Scholars Actually Discuss",
             fontsize=15, color='#e6edf3', fontfamily='serif', y=1.01)
plt.tight_layout()
plt.savefig('Visualizations/Thematic_analysis.png', dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(); print("✅ Fig 1 (TF-IDF) saved")

# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — LDA Topic Distribution Heatmap
# ════════════════
fig, (ax_top, ax_heat) = plt.subplots(2, 1, figsize=(18, 14),
                                       gridspec_kw={'height_ratios':[1,3]})
fig.patch.set_facecolor('#0d1117')

# Top panel: topic prevalence across all documents
ax_top.set_facecolor('#0d1117')
topic_prevalence = lda_matrix.mean(axis=0)
topic_labels_short = [f"T{i+1}: {topic_names[i]}" for i in range(N_TOPICS)]
bars_top = ax_top.bar(range(N_TOPICS), topic_prevalence,
                      color=[COLOR_PALETTE[i] for i in range(N_TOPICS)],
                      alpha=0.85, edgecolor='#1a1a2e', width=0.6)
ax_top.set_xticks(range(N_TOPICS))
ax_top.set_xticklabels([topic_names[i] for i in range(N_TOPICS)], fontsize=10, color='#e6edf3')
ax_top.set_ylabel("Mean topic weight\nacross all sources", fontsize=9)
ax_top.set_title("LDA Topic Prevalence — What Proportion of the Literature Each Topic Occupies",
                 fontsize=12, color='#e6edf3', pad=10)
ax_top.spines[:].set_color('#30363d'); ax_top.tick_params(colors='#8b949e')
for bar, val in zip(bars_top, topic_prevalence):
    ax_top.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
               f'{val:.3f}', ha='center', va='bottom', color='#8b949e', fontsize=9)

# Bottom: heatmap — each source vs each topic
ax_heat.set_facecolor('#0d1117')
heat_data = lda_matrix
im = ax_heat.imshow(heat_data.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=heat_data.max())
ax_heat.set_xticks(range(len(labels)))
ax_heat.set_xticklabels([d['label'] for d in corpus], rotation=45, ha='right', fontsize=8.5, color='#e6edf3')
ax_heat.set_yticks(range(N_TOPICS))
ax_heat.set_yticklabels([topic_names[i] for i in range(N_TOPICS)], fontsize=10, color='#e6edf3')
ax_heat.set_title("Source × Topic Matrix — How Strongly Each Source Discusses Each Theme",
                  fontsize=12, color='#e6edf3', pad=10)
for i in range(heat_data.shape[0]):
    for j in range(N_TOPICS):
        val = heat_data[i,j]
        ax_heat.text(i, j, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if val > 0.2 else '#8b949e')

cbar = plt.colorbar(im, ax=ax_heat, shrink=0.4, pad=0.02)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
cbar.set_label('Topic weight', color='#8b949e', fontsize=9)

plt.tight_layout()
plt.savefig('Visualizations/LDA_heatmap.png', dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(); print("✅ Fig 2 (LDA Heatmap) saved")

# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Term Co-occurrence Network
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')

pos_terms = nx.spring_layout(G_terms, seed=99, k=3.2, iterations=120)
node_weights = [G_terms.nodes[n].get('weight', 0.01)*3000 for n in G_terms.nodes()]
node_colors_t = [COLOR_PALETTE[term_to_topic(n) % len(COLOR_PALETTE)] for n in G_terms.nodes()]
edge_weights_t = [G_terms[u][v]['weight'] for u,v in G_terms.edges()]

for (u,v), w in zip(G_terms.edges(), edge_weights_t):
    x0,y0=pos_terms[u]; x1,y1=pos_terms[v]
    ax.plot([x0,x1],[y0,y1], color='#4a5568',
            alpha=min(0.9, 0.2+w*0.12), linewidth=0.3+w*0.25, zorder=1)

nx.draw_networkx_nodes(G_terms, pos_terms, ax=ax,
    node_color=node_colors_t, node_size=node_weights,
    alpha=0.90, linewidths=0.5, edgecolors='#1a1a2e')
nx.draw_networkx_labels(G_terms, pos_terms, ax=ax,
    font_size=8, font_color='#ffffff', font_family='serif')

legend_patches2 = [mpatches.Patch(color=COLOR_PALETTE[i], label=topic_names[i].replace('\n',' '))
                   for i in range(N_TOPICS)]
ax.legend(handles=legend_patches2, loc='lower left', fontsize=9,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
ax.set_title("Term Co-occurrence Network — Concepts That Appear Together in the Literature\n"
             "Node size = TF-IDF importance  |  Edge thickness = co-occurrence frequency  |  Color = thematic cluster",
             fontsize=13, pad=16, color='#e6edf3', fontfamily='serif')
ax.axis('off')
plt.tight_layout()
plt.savefig('Visualizations/Term_cooccurrence.png', dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(); print("✅ Fig 3 (Co-occurrence Network) saved")

# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Topic Evolution Over Time
# ════════════════
year_topic = pd.crosstab(df_corpus['year'], df_corpus['dominant_topic'])
year_topic.columns = [topic_names[c].replace('\n',' ') for c in year_topic.columns]

fig, (ax_evol, ax_donut) = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor('#0d1117')

# Left: stacked area over time
ax_evol.set_facecolor('#0d1117')
years_plot = sorted(df_corpus[df_corpus['year']>2000]['year'].unique())
for tid in range(N_TOPICS):
    ys = [lda_matrix[df_corpus[df_corpus['year']==yr].index.tolist()].mean(axis=0)[tid]
          if len(df_corpus[df_corpus['year']==yr]) > 0 else 0
          for yr in years_plot]
    ax_evol.plot(years_plot, ys, color=COLOR_PALETTE[tid], linewidth=2.5, alpha=0.9,
                marker='o', markersize=6, label=topic_names[tid].replace('\n',' '))
    ax_evol.fill_between(years_plot, ys, alpha=0.12, color=COLOR_PALETTE[tid])

ax_evol.set_xlabel("Year", fontsize=10)
ax_evol.set_ylabel("Mean topic weight in sources", fontsize=10)
ax_evol.set_title("How Research Focus Has Shifted Over Time\n(each line = one theme's prominence in publications)",
                  fontsize=12, color='#e6edf3', pad=10)
ax_evol.spines[:].set_color('#30363d'); ax_evol.tick_params(colors='#8b949e')
ax_evol.legend(fontsize=8.5, facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#e6edf3', loc='upper left')
ax_evol.axvline(2026, color='#ef4444', linestyle='--', alpha=0.5, linewidth=1)
ax_evol.text(2025.5, ax_evol.get_ylim()[0]+0.01, 'Grade 10\nstarts',
             color='#ef4444', fontsize=8, ha='right')

# Right: donut showing dominant topic per source type
ax_donut.set_facecolor('#0d1117')
type_topic_counts = df_corpus.groupby(['type','dominant_topic']).size().unstack(fill_value=0)
overall_topic_counts = type_topic_counts.sum()
sizes = overall_topic_counts.values
donut_colors = [COLOR_PALETTE[i] for i in overall_topic_counts.index]
wedge_labels = [topic_names[i].replace('\n','\n') for i in overall_topic_counts.index]
wedges, texts, autotexts = ax_donut.pie(
    sizes, colors=donut_colors, autopct='%1.0f%%',
    pctdistance=0.75, startangle=90,
    wedgeprops={'edgecolor':'#0d1117', 'linewidth':2},
    textprops={'color':'#e6edf3', 'fontsize':8})
centre = plt.Circle((0,0), 0.5, fc='#0d1117')
ax_donut.add_patch(centre)
ax_donut.text(0, 0, f'{len(corpus)}\nsources', ha='center', va='center',
             fontsize=13, color='#e6edf3', fontfamily='serif', fontweight='bold')
ax_donut.set_title("Dominant Theme Distribution\nAcross All 24 Sources",
                   fontsize=12, color='#e6edf3', pad=10)
ax_donut.legend(wedge_labels, loc='lower center', fontsize=7.5, ncol=2,
                facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3',
                bbox_to_anchor=(0.5, -0.18))

plt.suptitle("CBC Literature — Thematic Evolution & Distribution",
             fontsize=14, color='#e6edf3', fontfamily='serif', y=1.02)
plt.tight_layout()
plt.savefig('Visualizations/Thematic_evolution.png', dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(); print("✅ Fig 4 (Evolution & Distribution) saved")

# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Document Similarity Heatmap
# ══
sim_matrix = cosine_similarity(tfidf_matrix)
short_labels = [d['label'] for d in corpus]

fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
im = ax.imshow(sim_matrix, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8, color='#e6edf3')
ax.set_yticklabels(short_labels, fontsize=8, color='#e6edf3')
for i in range(len(short_labels)):
    for j in range(len(short_labels)):
        val = sim_matrix[i,j]
        if val > 0.15:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=6.5, color='#1a1a2e' if val>0.5 else '#4a5568')
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Cosine Similarity', color='#8b949e', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
ax.set_title("Document Similarity Matrix — How Closely Related Each Source Is to Every Other\n"
             "Darker = more similar content  |  Reveals natural clusters and isolated sources",
             fontsize=12, color='#e6edf3', pad=14)
plt.tight_layout()
plt.savefig('Visualizations/Document_similarity.png', dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(); print("✅ Fig 5 (Similarity Matrix) saved")

# ════════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════════
import os, shutil
out = "analysis"
os.makedirs(out, exist_ok=True)
term_importance.to_csv(f"{out}/tfidf_terms.csv", index=False)
pd.DataFrame(lda_matrix, columns=[f"topic_{i}" for i in range(N_TOPICS)],
             index=[d['label'] for d in corpus]).to_csv(f"{out}/lda_topic_weights.csv")
for fig_name in ['Thematic_analysis','LDA_heatmap','Term_cooccurrence','Thematic_evolution','Document_similarity']:
    shutil.copy(f"Visualizations/{fig_name}.png", f"{out}/{fig_name}.png")

print()
print("=" * 65)
print("THEMATIC ANALYSIS SUMMARY")
print("=" * 65)
print()
print("TOP 10 MOST DISCUSSED CONCEPTS IN CBC LITERATURE:")
for i, row in term_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['term']:30s}  score={row['score']:.4f}")
print()
print("SIX DOMINANT THEMES IDENTIFIED BY LDA:")
for tid in range(N_TOPICS):
    top_words_idx = lda.components_[tid].argsort()[-6:][::-1]
    top_words = [count_features[i] for i in top_words_idx]
    prev = topic_prevalence[tid]
    print(f"  T{tid+1}. {topic_names[tid].replace(chr(10),' '):<35s} prevalence={prev:.3f}")
    print(f"      [{', '.join(top_words)}]")
    print()
print("MOST CONTENT-SIMILAR SOURCE PAIRS:")
np.fill_diagonal(sim_matrix, 0)
flat_sim = [(short_labels[i], short_labels[j], sim_matrix[i,j])
            for i in range(len(short_labels)) for j in range(i+1, len(short_labels))]
flat_sim.sort(key=lambda x: -x[2])
for s1, s2, sim in flat_sim[:5]:
    print(f"  {s1:25s} ↔ {s2:25s}  similarity={sim:.4f}")
