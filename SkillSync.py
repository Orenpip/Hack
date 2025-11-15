#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install spacy


# In[7]:


# ============================================================================
# COURSE SYLLABI + RESUME KEYWORD EXTRACTION PIPELINE
# For Canvas-Career Bridge Matching System
# ============================================================================
import pandas as pd
import re
import json
from typing import List, Dict, Set, Tuple
import spacy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Optional: Install if needed
# !pip install spacy yake-keyword pandas
# !python -m spacy download en_core_web_sm


class CourseResumeKeywordExtractor:
    """
    Extract and normalize keywords from course descriptions and resumes.
    Uses spaCy for NLP, with n-gram generation and abbreviation expansion.
    """
    
    def __init__(self):
        """Initialize NLP models and lookup dictionaries."""
        
        print("Initializing Course & Resume Keyword Extractor...")
        
        # Load spaCy model for NLP
        print("  Loading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("  Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # ====================================================================
        # STOPWORDS
        # ====================================================================
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'we', 'you', 'they', 'them', 'their', 'our', 'your', 'my', 'me', 'i',
            'he', 'she', 'it', 'who', 'what', 'where', 'when', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'course', 'student', 'students', 'class', 'semester',
            'week', 'weeks', 'include', 'includes', 'including', 'also', 'well',
            'use', 'using', 'used', 'learn', 'learning', 'introduce', 'introduction'
        }
        
        # ====================================================================
        # ABBREVIATION EXPANSION DICTIONARY
        # ====================================================================
        self.abbreviation_map = {
            # Machine Learning & AI
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning',
            'nn': 'neural networks',
            'cnn': 'convolutional neural networks',
            'rnn': 'recurrent neural networks',
            'gan': 'generative adversarial networks',
            
            # Programming & Development
            'oop': 'object oriented programming',
            'api': 'application programming interface',
            'rest': 'representational state transfer',
            'crud': 'create read update delete',
            'mvc': 'model view controller',
            'ui': 'user interface',
            'ux': 'user experience',
            'sdk': 'software development kit',
            'ide': 'integrated development environment',
            
            # Data & Databases
            'sql': 'structured query language',
            'nosql': 'non-relational database',
            'rdbms': 'relational database management system',
            'etl': 'extract transform load',
            'olap': 'online analytical processing',
            'oltp': 'online transaction processing',
            'bi': 'business intelligence',
            'eda': 'exploratory data analysis',
            
            # Statistics & Analysis
            'anova': 'analysis of variance',
            'regression': 'regression analysis',
            'pca': 'principal component analysis',
            'svm': 'support vector machine',
            'knn': 'k nearest neighbors',
            'rf': 'random forest',
            
            # Cloud & DevOps
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'cicd': 'continuous integration continuous deployment',
            'ci/cd': 'continuous integration continuous deployment',
            'vm': 'virtual machine',
            
            # Business & Management
            'crm': 'customer relationship management',
            'erp': 'enterprise resource planning',
            'roi': 'return on investment',
            'kpi': 'key performance indicator',
            'b2b': 'business to business',
            'b2c': 'business to consumer',
            'saas': 'software as a service',
            'paas': 'platform as a service',
            'iaas': 'infrastructure as a service',
            
            # Academic & Research
            'apa': 'american psychological association',
            'mla': 'modern language association',
            'gpa': 'grade point average',
            'stem': 'science technology engineering mathematics',
            
            # Other
            'html': 'hypertext markup language',
            'css': 'cascading style sheets',
            'xml': 'extensible markup language',
            'json': 'javascript object notation',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'url': 'uniform resource locator',
            'gui': 'graphical user interface',
            'cli': 'command line interface',
            'os': 'operating system',
            'io': 'input output',
            'ar': 'augmented reality',
            'vr': 'virtual reality',
            'iot': 'internet of things',
            'gis': 'geographic information system'
        }
        
        # ====================================================================
        # SYNONYM NORMALIZATION
        # ====================================================================
        self.synonym_map = {
            # Programming synonyms
            'coding': 'programming',
            'software development': 'programming',
            'software engineering': 'programming',
            'scripting': 'programming',
            
            # Data synonyms
            'data science': 'data analysis',
            'analytics': 'data analysis',
            'data analytics': 'data analysis',
            'statistical analysis': 'statistics',
            'statistical methods': 'statistics',
            'quantitative analysis': 'statistics',
            'quantitative methods': 'statistics',
            
            # Database synonyms
            'database management': 'database',
            'data storage': 'database',
            'data warehouse': 'database',
            
            # Modeling synonyms
            'predictive modeling': 'modeling',
            'statistical modeling': 'modeling',
            'mathematical modeling': 'modeling',
            
            # Visualization synonyms
            'data visualization': 'visualization',
            'visual analytics': 'visualization',
            'graphical analysis': 'visualization',
            
            # Research synonyms
            'research methods': 'research',
            'research design': 'research',
            'empirical research': 'research',
            
            # Analysis synonyms
            'econometric analysis': 'econometrics',
            'regression modeling': 'regression',
            'time series analysis': 'time series',
            
            # Communication synonyms
            'technical writing': 'writing',
            'business writing': 'writing',
            'oral presentation': 'presentation',
            'public speaking': 'presentation'
        }
        
        # ====================================================================
        # IMPORTANT SKILLS/CONCEPTS TO PRIORITIZE
        # ====================================================================
        self.important_terms = {
            # Technical skills
            'python', 'r', 'java', 'javascript', 'sql', 'c++', 'matlab',
            'tableau', 'excel', 'power bi', 'git', 'docker', 'kubernetes',
            
            # Methodologies
            'machine learning', 'deep learning', 'data analysis', 'statistics',
            'regression', 'hypothesis testing', 'statistical inference',
            'econometrics', 'time series', 'panel data', 'causal inference',
            
            # Domain concepts
            'optimization', 'simulation', 'modeling', 'forecasting',
            'algorithm', 'data structure', 'database', 'visualization',
            'research', 'experimentation', 'survey design', 'sampling',
            
            # Soft skills
            'communication', 'teamwork', 'leadership', 'problem solving',
            'critical thinking', 'project management', 'presentation',
            
            # Business concepts
            'supply chain', 'operations', 'finance', 'marketing', 'strategy',
            'policy analysis', 'economic analysis', 'business analysis'
        }
        
        print("✓ Extractor initialized\n")
    
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Steps:
        1. Convert to lowercase
        2. Remove HTML tags
        3. Remove special characters but keep spaces and hyphens
        4. Remove extra whitespace
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces, hyphens, and forward slashes
        text = re.sub(r'[^\w\s/-]', ' ', text)
        
        # Remove standalone numbers (but keep numbers within words like "cs101")
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and lemmatize using spaCy.
        
        Returns:
        - List of lemmatized tokens (excluding stopwords and short words)
        """
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip if it's a stopword, punctuation, or very short
            if (token.text.lower() in self.stopwords or 
                token.is_punct or 
                token.is_space or 
                len(token.text) < 2):
                continue
            
            # Use lemma (base form of word)
            lemma = token.lemma_.lower()
            
            # Skip if lemmatized form is a stopword
            if lemma not in self.stopwords:
                tokens.append(lemma)
        
        return tokens
    
    
    def generate_ngrams(self, tokens: List[str], max_n: int = 3) -> List[str]:
        """
        Generate n-grams (1-grams, 2-grams, 3-grams).
        
        Prioritizes academically meaningful phrases.
        """
        ngrams = []
        
        # Add unigrams
        ngrams.extend(tokens)
        
        # Add bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            ngrams.append(bigram)
        
        # Add trigrams
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            ngrams.append(trigram)
        
        return ngrams
    
    
    def expand_abbreviations(self, ngrams: List[str]) -> Tuple[List[str], List[str]]:
        """
        Expand abbreviations found in n-grams.
        
        Returns:
        - expanded_ngrams: n-grams with abbreviations expanded
        - expansions_found: list of (abbrev, expansion) pairs found
        """
        expanded_ngrams = []
        expansions_found = []
        
        for ngram in ngrams:
            if ngram in self.abbreviation_map:
                # Found an abbreviation
                expansion = self.abbreviation_map[ngram]
                expanded_ngrams.append(expansion)
                expansions_found.append(f"{ngram} → {expansion}")
                # Also keep the original abbreviation
                expanded_ngrams.append(ngram)
            else:
                expanded_ngrams.append(ngram)
        
        return expanded_ngrams, expansions_found
    
    
    def normalize_synonyms(self, ngrams: List[str]) -> List[str]:
        """
        Normalize synonyms to canonical forms.
        """
        normalized = []
        
        for ngram in ngrams:
            if ngram in self.synonym_map:
                canonical = self.synonym_map[ngram]
                normalized.append(canonical)
            else:
                normalized.append(ngram)
        
        return normalized
    
    
    def extract_final_keywords(self, ngrams: List[str], top_n: int = 50) -> List[str]:
        """
        Extract final keywords by:
        1. Removing duplicates
        2. Prioritizing important terms
        3. Filtering by frequency
        4. Preferring longer phrases
        """
        # Count frequencies
        ngram_counts = Counter(ngrams)
        
        # Separate into important and other
        important_keywords = []
        other_keywords = []
        
        for ngram, count in ngram_counts.items():
            if ngram in self.important_terms:
                important_keywords.append((ngram, count, len(ngram.split())))
            else:
                other_keywords.append((ngram, count, len(ngram.split())))
        
        # Sort important keywords by: length (longer = better), then frequency
        important_keywords.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Sort other keywords similarly
        other_keywords.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Combine: prioritize important terms
        final_keywords = (
            [kw[0] for kw in important_keywords] + 
            [kw[0] for kw in other_keywords]
        )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in final_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:top_n]
    
    
    def process_text(self, text: str, text_type: str = "course") -> Dict:
        """
        Complete pipeline to process text and extract keywords.
        
        Args:
            text: Input text (course description or resume)
            text_type: "course" or "resume" (for logging)
            
        Returns:
            Dictionary with all intermediate and final results
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        # Step 3: Generate n-grams
        ngrams = self.generate_ngrams(tokens, max_n=3)
        
        # Step 4: Expand abbreviations
        expanded_ngrams, expansions = self.expand_abbreviations(ngrams)
        
        # Step 5: Normalize synonyms
        normalized_ngrams = self.normalize_synonyms(expanded_ngrams)
        
        # Step 6: Extract final keywords
        final_keywords = self.extract_final_keywords(normalized_ngrams, top_n=50)
        
        return {
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "ngrams": ngrams[:20],  # Sample for debugging
            "expanded_abbreviations": expansions,
            "final_keywords": final_keywords
        }
    
    
    def process_courses_csv(self, csv_path: str, 
                           course_name_col: str = 'course_name',
                           course_desc_col: str = 'course_description',
                           resume_col: str = 'resume') -> Dict:
        """
        Process CSV file containing courses and resume.
        
        Args:
            csv_path: Path to CSV file
            course_name_col: Column name for course names
            course_desc_col: Column name for course descriptions
            resume_col: Column name for resume text
            
        Returns:
            Complete structured output with all keywords
        """
        print("="*80)
        print("PROCESSING COURSES + RESUME CSV")
        print("="*80 + "\n")
        
        # Read CSV
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Found {len(df)} courses\n")
        
        # ====================================================================
        # PROCESS COURSES
        # ====================================================================
        print("Processing course descriptions...")
        
        courses_output = []
        all_course_keywords = []
        
        for idx, row in df.iterrows():
            course_name = row[course_name_col]
            course_desc = row[course_desc_col]
            
            print(f"  [{idx+1}/{len(df)}] {course_name}")
            
            # Process this course description
            result = self.process_text(course_desc, text_type="course")
            
            # Store course keywords
            courses_output.append({
                "course_name": course_name,
                "keywords": result["final_keywords"]
            })
            
            # Add to master list
            all_course_keywords.extend(result["final_keywords"])
        
        print(f"\n✓ Processed {len(courses_output)} courses\n")
        
        # ====================================================================
        # PROCESS RESUME (only once since it's the same in all rows)
        # ====================================================================
        print("Processing resume...")
        
        resume_text = df[resume_col].iloc[0]  # Get from first row
        resume_result = self.process_text(resume_text, text_type="resume")
        resume_keywords = resume_result["final_keywords"]
        
        print(f"  ✓ Extracted {len(resume_keywords)} resume keywords\n")
        
        # ====================================================================
        # CREATE UNIFIED MASTER LIST
        # ====================================================================
        print("Creating unified keyword master list...")
        
        # Combine all keywords
        all_keywords_combined = all_course_keywords + resume_keywords
        
        # Remove duplicates while preserving importance
        all_keywords_unique = self.extract_final_keywords(all_keywords_combined, top_n=100)
        
        print(f"  ✓ Unified list contains {len(all_keywords_unique)} unique keywords\n")
        
        # ====================================================================
        # BUILD FINAL OUTPUT
        # ====================================================================
        output = {
            "courses": courses_output,
            "resume_keywords": resume_keywords,
            "all_keywords": all_keywords_unique,
            "statistics": {
                "total_courses": len(courses_output),
                "total_course_keywords": len(all_course_keywords),
                "total_resume_keywords": len(resume_keywords),
                "total_unique_keywords": len(all_keywords_unique)
            }
        }
        
        print("="*80)
        print("PROCESSING COMPLETE")
        print("="*80 + "\n")
        
        return output


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize extractor
    extractor = CourseResumeKeywordExtractor()
    
    # ========================================================================
    # CREATE SAMPLE CSV FOR DEMONSTRATION
    # ========================================================================
    
    sample_data = pd.DataFrame({
        'course_name': [
            'STAT 215 - Statistical Inference',
            'CS 101 - Introduction to Programming',
            'ECON 301 - Applied Econometrics',
            'EDUC 200 - Education Policy'
        ],
        'course_description': [
            'Introduction to statistical inference including hypothesis testing, confidence intervals, and regression analysis. Students will learn to apply statistical methods to real-world data using R programming. Topics include ANOVA, linear regression, and experimental design.',
            'Fundamental programming concepts using Python. Topics include data structures, algorithms, OOP principles, and API development. Students will build projects using HTML, CSS, and JavaScript.',
            'Application of statistical and econometric methods to economic data. Focus on regression models, panel data analysis, time series, and causal inference techniques. Use of Stata and R for analysis.',
            'Examination of contemporary education policy issues including equity, access, and educational outcomes. Analysis of policy interventions using data-driven approaches and program evaluation methods.'
        ],
        'resume': [
            # Same resume in all rows
            '''John Doe
            Education: Bachelor of Science in Statistics, Minor in Computer Science
            
            Skills:
            - Programming: Python, R, SQL, Java
            - Data Analysis: Statistical Analysis, ML, Data Visualization, Tableau
            - Tools: Excel, Git, AWS
            
            Experience:
            Research Assistant | Dept of Education | 2024-Present
            - Statistical analysis using R and Python
            - Created visualizations using Tableau
            - ML models for predictive analytics
            
            Data Science Club President | 2023-Present
            - Led workshops on data analysis
            - Organized hackathons
            
            Soft Skills: Communication, Teamwork, Leadership, Problem Solving'''
        ] * 4  # Repeat same resume for all rows
    })
    
    # Save to CSV
    sample_data.to_csv('sample_courses.csv', index=False)
    print("Created sample CSV: sample_courses.csv\n")
    
    # ========================================================================
    # PROCESS THE CSV
    # ========================================================================
    
    results = extractor.process_courses_csv(
        'sample_courses.csv',
        course_name_col='course_name',
        course_desc_col='course_description',
        resume_col='resume'
    )
    
    # ========================================================================
    # SAVE TO JSON
    # ========================================================================
    
    with open('extracted_keywords.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Saved results to: extracted_keywords.json\n")
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"Statistics:")
    for key, value in results['statistics'].items():
        print(f"  • {key}: {value}")
    
    print("\n" + "-"*80 + "\n")
    
    print("Course Keywords (sample):")
    for course in results['courses'][:2]:  # Show first 2
        print(f"\n{course['course_name']}:")
        print(f"  Keywords: {', '.join(course['keywords'][:10])}...")
    
    print("\n" + "-"*80 + "\n")
    
    print("Resume Keywords (first 20):")
    print(f"  {', '.join(results['resume_keywords'][:20])}")
    
    print("\n" + "-"*80 + "\n")
    
    print("Unified Master List (first 30):")
    print(f"  {', '.join(results['all_keywords'][:30])}")
    
    print("\n" + "="*80)

