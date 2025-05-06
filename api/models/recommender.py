import argparse
import os
import json
from typing import List, Dict, Tuple
from transformers import BertTokenizer # BertModel removed
# import torch # torch removed
from groq import Groq
from rank_bm25 import BM25Okapi
import re
import logging
from dotenv import load_dotenv
from functools import lru_cache # Added
from collections import defaultdict # Added

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHLRecommender:
    def __init__(self):
        """
        Initializes the SHLRecommender with assessments, tokenizer,
        BM25 index, Groq client, and enhanced skill mappings.
        """
        self.assessments = self._load_assessments()
        
        try:
            # Tokenizer might still be useful for _prepare_bm25_data or future non-BERT embeddings
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.warning(f"Could not load BERT tokenizer. Some tokenization might be affected: {e}")
            self.tokenizer = None # Keep tokenizer for BM25's tokenization if no other is specified

        self.bm25 = None
        self.tokenized_corpus_for_bm25 = []
        self._prepare_bm25_data() # Uses self.tokenizer if available, otherwise basic regex
        
        self._enhance_technical_skills() # New method call

        try:
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
            if not self.groq_api_key:
                logger.warning("GROQ_API_KEY not found in environment variables. LLM features will be disabled.")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=self.groq_api_key)
            self.llm_model = "llama3-70b-8192"
            self.fast_llm_model = "llama3-8b-8192"
            logger.info("Groq client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.groq_client = None

    def _enhance_technical_skills(self):
        """
        Defines mappings for specific technical skills and pre-calculates
        which assessments relate to them based on keywords in their descriptions.
        This also addresses specific error analysis feedback.
        """
        self.tech_skill_map = {
            'selenium': {'automated testing', 'web testing', 'test automation', 'qa automation'},
            'manual testing': {'qa testing', 'test cases', 'quality assurance', 'manual qa', 'test execution', 'test case management'},
            'cultural fit': {'cross-cultural', 'global mindset', 'values alignment', 'intercultural competence', 'global leadership'},
            # Add more mappings as needed
        }
        logger.info(f"Initialized tech_skill_map with keys: {list(self.tech_skill_map.keys())}")

        self.skill_to_assessments = defaultdict(set)
        if not self.assessments:
            logger.warning("No assessments loaded, cannot populate skill_to_assessments.")
            return

        for assessment in self.assessments:
            assessment_name = assessment.get('name', '')
            description = assessment.get('description', '').lower()
            assessment_skills_raw = assessment.get('skills', [])
            assessment_skills = [str(s).lower() for s in assessment_skills_raw if isinstance(s, str)]


            for skill_key, keywords in self.tech_skill_map.items():
                # An assessment is related to skill_key if its description or skills contain any of the keywords,
                # OR if its name/description/skills directly contain the skill_key itself.
                matches_skill_key_directly = skill_key in description or skill_key in assessment_name.lower() or skill_key in assessment_skills
                matches_any_keyword = any(kw in description for kw in keywords) or \
                                      any(kw in ' '.join(assessment_skills) for kw in keywords)
                
                if matches_skill_key_directly or matches_any_keyword:
                    self.skill_to_assessments[skill_key].add(assessment_name)
        
        count_mapped = sum(len(v) for v in self.skill_to_assessments.values())
        logger.info(f"Populated skill_to_assessments: {len(self.skill_to_assessments)} skill keys mapped, total {count_mapped} assessment name associations.")


    def _load_assessments(self) -> List[Dict]:
        """
        Loads assessments from a JSON file.
        Adjusted path logic.
        """
        try:
            # Try to find the file relative to the current script's directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Potential paths, common project structures:
            # 1. ../data/file.json (data folder is sibling to script's parent folder)
            # 2. ../../data/file.json (data folder is sibling to script's grandparent folder - common for scripts in src/models)
            # 3. ./data/file.json (data folder is sibling to script itself)
            # 4. Default path from original complex logic (project_root_assumed / data / file.json)
            
            filename = 'assessment_embeddings_groq_v2_BERT.json'
            potential_paths_to_check = [
                os.path.join(current_script_dir, '..', 'data', filename),
                os.path.join(current_script_dir, '..', '..', 'data', filename),
                os.path.join(current_script_dir, 'data', filename),
            ]
            
            # Add the original assumed path structure as a fallback
            project_root_assumed = os.path.dirname(os.path.dirname(current_script_dir)) # project_root/api/models -> project_root
            original_assumed_path = os.path.join(project_root_assumed, 'data', filename)
            if original_assumed_path not in [os.path.abspath(p) for p in potential_paths_to_check]: # Avoid duplicate checks
                 potential_paths_to_check.append(original_assumed_path)


            data_path = None
            for p_path in potential_paths_to_check:
                abs_path = os.path.abspath(p_path)
                logger.debug(f"Attempting to load assessments from: {abs_path}")
                if os.path.exists(abs_path):
                    data_path = abs_path
                    logger.info(f"Found assessment file at: {data_path}")
                    break
            
            if not data_path:
                logger.error(f"Assessment file '{filename}' not found in any of the checked locations: {potential_paths_to_check}")
                return []

            with open(data_path, 'r', encoding='utf-8') as f:
                assessments = json.load(f)
            logger.info(f"Loaded {len(assessments)} assessments from {data_path}.")
            return assessments
        except Exception as e:
            logger.error(f"Failed to load assessments: {e}")
            return []

    def _prepare_bm25_data(self):
        if not self.assessments:
            logger.warning("No assessments loaded, BM25 preparation skipped.")
            return
        corpus_for_bm25 = []
        for assessment in self.assessments:
            domain_text = assessment.get('domain', "")
            if isinstance(domain_text, list): domain_text = ", ".join(map(str,domain_text)) # Ensure all elements are strings
            else: domain_text = str(domain_text)

            skills_list = assessment.get('skills', [])
            skills_text = " ".join([str(s) for s in skills_list if isinstance(s, str)])


            text_parts = [
                str(assessment.get('name', '')),
                str(assessment.get('description', '')),
                skills_text,
                domain_text
            ]
            full_text = " ".join(filter(None, text_parts))
            
            # Use BERT tokenizer if available for potentially better tokenization, else regex
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(full_text.lower())
            else:
                tokens = re.findall(r'\b\w+\b', full_text.lower())
            corpus_for_bm25.append(tokens)

        if corpus_for_bm25:
            self.bm25 = BM25Okapi(corpus_for_bm25)
            self.tokenized_corpus_for_bm25 = corpus_for_bm25 # Storing for potential inspection
            logger.info(f"BM25 index prepared with {len(corpus_for_bm25)} documents.")
        else:
            logger.warning("Corpus for BM25 is empty. BM25 index not created.")

    def extract_query_criteria(self, query: str) -> Dict:
        if not self.groq_client:
            logger.warning("Groq client not available. Using fallback for criteria extraction.")
            domain, skills = self._fallback_domain_skills(query)
            cultural_context_fb = self._detect_cultural_focus(query) # Detect cultural focus even in fallback
            return {"domain": domain, "skills": skills, "duration_minutes": None, "experience_level": None, "raw_query": query, "cultural_context": cultural_context_fb}

        # Detect cultural context first to potentially inform the LLM during extraction
        cultural_context = self._detect_cultural_focus(query)

        prompt = f"""Analyze the following job description to extract structured information.
Job Description: "{query}"
{"Identified cultural context clues (for your awareness, do not list these in skills unless they are actual job skills): " + ", ".join(cultural_context) if cultural_context else ""}

Extract the following:
1.  "domain": The primary job domain (e.g., "Software Engineering", "Sales", "Quality Assurance"). If multiple are implied, pick the most central one.
2.  "skills": A list of 3-5 most relevant technical or soft skills (e.g., ["java", "collaboration", "sales techniques", "manual testing", "selenium"]). Prioritize skills directly mentioned or strongly implied.
3.  "duration_minutes": If a desired assessment completion time is mentioned (e.g., "around 30 minutes", "less than 40 minutes", "up to 1 hour"), extract the maximum duration in minutes as an integer. If a range is given (e.g., "30-45 minutes"), use the upper bound. If no specific duration is mentioned, return null.
4.  "experience_level": If an experience level is mentioned (e.g., "new graduate", "senior developer", "entry-level sales"), classify it into one of: "entry", "junior", "mid", "senior", "any". If not mentioned, return null.

Return ONLY a valid JSON object with the keys "domain", "skills", "duration_minutes", and "experience_level".
The value for "skills" should be an array of strings.
The value for "duration_minutes" should be an integer or null.
The value for "experience_level" should be a string or null.
"""
        content = "" 
        try:
            response = self.groq_client.chat.completions.create(
                model=self.fast_llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            parsed = json.loads(content.strip())
            
            domain = parsed.get("domain", "").strip().lower()
            skills_raw = parsed.get("skills", [])
            if not isinstance(skills_raw, list): skills_raw = [skills_raw] if skills_raw else []
            skills = list(dict.fromkeys([s.strip().lower() for s in skills_raw if isinstance(s, str) and s.strip()]))
            
            duration = parsed.get("duration_minutes")
            if duration is not None:
                try:
                    duration = int(duration)
                except (ValueError, TypeError):
                    logger.warning(f"LLM returned non-integer duration: {duration}. Setting to None.")
                    duration = None
            
            experience = parsed.get("experience_level")
            if experience and isinstance(experience, str):
                experience = experience.strip().lower()
                if not experience: experience = None
            else: 
                experience = None

            if not domain and not skills:
                logger.warning("LLM extraction resulted in empty domain and skills. Using fallback for domain/skills.")
                domain_fb, skills_fb = self._fallback_domain_skills(query)
                domain = domain_fb
                skills = skills_fb
            elif not domain and skills:
                 domain = self._infer_domain_from_skills_or_query(skills, query)

            return {
                "domain": domain, 
                "skills": skills[:5],
                "duration_minutes": duration,
                "experience_level": experience,
                "raw_query": query,
                "cultural_context": cultural_context # Add detected cultural context
            }
        except Exception as e:
            logger.error(f"LLM criteria extraction failed: {e}. Content: '{content}'. Using fallback.")
            domain_fb, skills_fb = self._fallback_domain_skills(query)
            return {"domain": domain_fb, "skills": skills_fb, "duration_minutes": None, "experience_level": None, "raw_query": query, "cultural_context": cultural_context}

    def _infer_domain_from_skills_or_query(self, skills: List[str], query: str) -> str:
        combined_text = " ".join(skills) + " " + query.lower()
        if "java" in combined_text or "spring" in combined_text: return "java development"
        if "python" in combined_text or "django" in combined_text or "flask" in combined_text: return "python development"
        if "javascript" in combined_text or "react" in combined_text or "angular" in combined_text or "node.js" in combined_text: return "web development"
        if "sql" in combined_text or "database" in combined_text: return "data management"
        if "sales" in combined_text or "crm" in combined_text: return "sales"
        if "customer service" in combined_text or "support" in combined_text: return "customer support"
        if "project manag" in combined_text: return "project management"
        if "manual test" in combined_text or "qa test" in combined_text or "quality assurance" in combined_text: return "quality assurance"
        if "selenium" in combined_text or "automated test" in combined_text: return "test automation"
        return "general technical"

    def _fallback_domain_skills(self, query: str) -> Tuple[str, List[str]]:
        query_lower = query.lower()
        domain_map = {
            'java': 'java development', 'spring boot': 'java development',
            'python': 'python development', 'django': 'python development', 'flask': 'python development',
            'javascript': 'web development', 'react': 'web development', 'angular': 'web development', 'vue': 'web development', 'node.js': 'web development',
            'c#': '.net development', '.net': '.net development', 'c++': 'c++ development',
            'ruby': 'ruby development', 'rails': 'ruby development', 'php': 'php development', 'laravel': 'php development',
            'swift': 'mobile development (ios)', 'kotlin': 'mobile development (android)', 'ios': 'mobile development (ios)', 'android': 'mobile development (android)',
            'sql': 'database management', 'database': 'database management',
            'cloud': 'cloud computing', 'aws': 'cloud computing', 'azure': 'cloud computing', 'gcp': 'cloud computing',
            'devops': 'devops', 'kubernetes': 'devops', 'docker': 'devops',
            'machine learning': 'machine learning', 'ai': 'artificial intelligence', 'data science': 'data science',
            'cybersecurity': 'cybersecurity', 'security': 'cybersecurity', 'sales': 'sales',
            'marketing': 'marketing', 'customer service': 'customer support', 'support': 'customer support',
            'project manager': 'project management', 'product manager': 'product management',
            'hr': 'human resources', 'recruitment': 'human resources', 'finance': 'finance',
            'developer': 'software development', 'engineer': 'software engineering',
            'manual test': 'quality assurance', 'qa test': 'quality assurance', 'quality assurance': 'quality assurance',
            'selenium': 'test automation', 'automated test': 'test automation', 'test automation': 'test automation'
        }
        skill_patterns = [
            (r'\bjava\b', 'java'), (r'spring boot', 'spring boot'), (r'\bpython\b', 'python'), (r'django', 'django'),
            (r'javascript', 'javascript'), (r'react\.?js', 'react'), (r'angular\.?js', 'angular'), (r'node\.?js', 'node.js'),
            (r'c#', 'c#'), (r'\.net', '.net framework'), (r'c\+\+', 'c++'), (r'\bsql\b', 'sql'),
            (r'aws', 'aws'), (r'azure', 'microsoft azure'), (r'gcp', 'google cloud platform'),
            (r'docker', 'docker'), (r'kubernetes', 'kubernetes'), (r'git', 'git'),
            (r'machine learning', 'machine learning'), (r'data science', 'data science'),
            (r'agile', 'agile methodologies'), (r'scrum', 'scrum'),
            (r'problem[- ]?solving', 'problem solving'), (r'communicat\w+', 'communication'),
            (r'team\w*', 'teamwork'), (r'collab\w*', 'collaboration'), (r'leadership', 'leadership'),
            (r'manual test\w*', 'manual testing'), (r'selenium', 'selenium'), (r'test automation', 'test automation'),
            (r'qa\b', 'quality assurance'), (r'test case\w*', 'test case design')
        ]
        extracted_domain = 'general technical' 
        for kw, dom_name in domain_map.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                extracted_domain = dom_name
                break
        
        extracted_skills = set()
        for pattern, skill_name in skill_patterns:
            if re.search(pattern, query_lower):
                extracted_skills.add(skill_name)

        for skill_key, keywords in self.tech_skill_map.items():
            if skill_key in query_lower or any(kw in query_lower for kw in keywords):
                extracted_skills.add(skill_key) 

        if extracted_domain == 'java development' and not any(s in extracted_skills for s in ['java', 'spring boot']):
            extracted_skills.update(['java', 'object-oriented programming'])
        elif extracted_domain == 'software development' and not extracted_skills:
            extracted_skills.update(['programming fundamentals', 'problem solving'])
        elif extracted_domain == 'sales' and not extracted_skills:
             extracted_skills.update(['communication', 'persuasion', 'customer interaction'])
        elif (extracted_domain == 'quality assurance' or extracted_domain == 'test automation') and not extracted_skills:
             extracted_skills.update(['attention to detail', 'analytical skills'])

        if not extracted_skills: 
            extracted_skills = ['communication', 'problem solving', 'teamwork']
            
        return extracted_domain, list(extracted_skills)[:5]

    def _is_technical_role(self, query: str) -> bool:
        """Identifies if a query likely relates to a technical role."""
        tech_indicators = {
            'developer', 'engineer', 'programming', 'software', 'technical',
            'code', 'coding', 'algorithm', 'data', 'database', 'network',
            'qa', 'test', 'automation', 'cybersecurity', 'devops', 'cloud',
            'java', 'python', 'c++', 'c#', 'javascript', 'sql', 'scripting',
            'selenium', 'manual testing' # Added from tech_skill_map keys
        }
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in tech_indicators)

    def _boost_technical_matches_on_scores(self, scored_indices: List[Tuple[float, int]], query: str) -> List[Tuple[float, int]]:
        """
        Boosts scores of assessments that match technical skills mentioned in the query,
        leveraging the pre-computed self.skill_to_assessments map.
        """
        boosted_scored_indices = []
        query_lower = query.lower()
        tech_skill_boost_value = 0.5 

        for score, assessment_idx in scored_indices:
            current_score_for_assessment = score # Keep track of original score for this item for logging
            boost_applied_this_assessment = False
            assessment = self.assessments[assessment_idx]
            assessment_name = assessment.get('name', '')

            for skill_key, mapped_keywords in self.tech_skill_map.items():
                query_mentions_skill = skill_key in query_lower or \
                                     any(kw in query_lower for kw in mapped_keywords)
                
                if query_mentions_skill:
                    if assessment_name in self.skill_to_assessments.get(skill_key, set()):
                        score += tech_skill_boost_value 
                        boost_applied_this_assessment = True
                        # Apply boost only once per skill_key match to avoid over-boosting from multiple query keywords for the same skill
                        # However, an assessment can be boosted by multiple distinct skill_keys if it matches them all
            
            if boost_applied_this_assessment:
                 logger.debug(f"Boosting '{assessment_name}' due to technical skill match. Score: {current_score_for_assessment:.2f} -> {score:.2f}")
            boosted_scored_indices.append((score, assessment_idx))
        return boosted_scored_indices

    def _adjust_for_duration(self, candidates: List[Dict], target_duration: int, pool_size: int) -> List[Dict]:
        """
        Sorts candidates by proximity to the target_duration and then takes the top 'pool_size'.
        Assessments without a duration are penalized (sorted towards the end).
        """
        if target_duration is None: 
            return candidates[:pool_size]

        def sort_key(assessment_dict):
            length_minutes_raw = assessment_dict.get('length_minutes')
            if length_minutes_raw is None:
                return (True, float('inf')) 
            try:
                length_minutes = int(length_minutes_raw)
                return (False, abs(length_minutes - target_duration))
            except (ValueError, TypeError): 
                logger.warning(f"Invalid duration '{length_minutes_raw}' for assessment '{assessment_dict.get('name')}' during sort. Penalizing.")
                return (True, float('inf')) 

        sorted_candidates = sorted(candidates, key=sort_key)
        
        logger.debug(f"Adjusted for duration: {target_duration} min. Sorted {len(candidates)} candidates, returning up to {pool_size}.")
        return sorted_candidates[:pool_size]

    @lru_cache(maxsize=256) 
    def recommend(self, query: str, top_k: int = 7, bm25_candidate_pool_size: int = 50) -> List[Dict]:
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Invalid query received. Query must be a non-empty string.")
            return []
        if not self.assessments:
            logger.error("No assessments available to recommend from.")
            return []
        if not self.bm25:
            logger.error("BM25 index not prepared. Cannot perform initial filtering.")
            return []
            
        logger.info(f"Processing query (first 100 chars): {query[:100]}...")
        criteria = self.extract_query_criteria(query) 
        domain = criteria["domain"]
        skills = criteria["skills"]
        duration_constraint = criteria["duration_minutes"] 

        logger.info(f"Extracted Criteria: Domain='{domain}', Skills={skills}, Max Duration='{duration_constraint}', Exp='{criteria['experience_level']}', CultCtx={criteria['cultural_context']}")
        
        if self.tokenizer:
            query_tokens_list = self.tokenizer.tokenize(query.lower())
        else:
            query_tokens_list = re.findall(r'\b\w+\b', query.lower())

        if not query_tokens_list and (domain or skills):
            logger.info("Query is generic or lacks keywords, augmenting with extracted domain/skills for BM25.")
            augment_terms = []
            if domain: augment_terms.extend(re.findall(r'\b\w+\b', domain.lower()))
            for skill_item in skills: # skills is a list of strings
                 augment_terms.extend(re.findall(r'\b\w+\b', skill_item.lower())) # skill_item is a string
            
            if self.tokenizer: 
                query_tokens_list.extend(self.tokenizer.tokenize(" ".join(augment_terms)))
            else:
                query_tokens_list.extend(augment_terms)
            query_tokens_list = list(dict.fromkeys(query_tokens_list))


        if not query_tokens_list:
            logger.warning("No effective tokens for BM25 search after processing query, domain, and skills.")
            return []

        bm25_scores = self.bm25.get_scores(query_tokens_list)
        
        scored_indices = []
        for i, score in enumerate(bm25_scores):
            # Ensure score is float, BM25Okapi sometimes returns numpy.float64
            current_bm25_score = float(score)

            assessment_domain_text_raw = self.assessments[i].get('domain', "")
            if isinstance(assessment_domain_text_raw, list): 
                assessment_domain_text = " ".join(map(str, assessment_domain_text_raw)).lower()
            else: 
                assessment_domain_text = str(assessment_domain_text_raw).lower()

            current_domain_boost = 1.0
            if domain and domain in assessment_domain_text:
                current_domain_boost = 1.5  
            elif domain and any(d_part in assessment_domain_text for d_part in domain.split()):
                 current_domain_boost = 1.2 
            
            assessment_skills_list_raw = self.assessments[i].get('skills', [])
            assessment_skills_lower = [str(s).lower() for s in assessment_skills_list_raw if isinstance(s, str)]
            
            skill_overlap_count = sum(1 for s_query in skills if s_query in assessment_skills_lower)
            current_skill_boost = 1.0 + (0.15 * skill_overlap_count)

            final_score = current_bm25_score * current_domain_boost * current_skill_boost
            if current_bm25_score > 0: 
                 scored_indices.append((final_score, i))
        
        if self._is_technical_role(query):
            logger.info("Query identified as technical role, applying technical skill boosts.")
            scored_indices = self._boost_technical_matches_on_scores(scored_indices, query)
                
        scored_indices.sort(reverse=True, key=lambda x: x[0])
        
        initial_bm25_pool_size = max(bm25_candidate_pool_size * 3, top_k * 10) 
        bm25_top_indices = [idx for _, idx in scored_indices[:initial_bm25_pool_size]]
        assessments_for_filtering = [self.assessments[i] for i in bm25_top_indices]
        logger.info(f"BM25 (+boosts) initially retrieved {len(assessments_for_filtering)} candidates before pre-rerank filtering.")

        pre_rerank_filtered_assessments = []
        for assessment in assessments_for_filtering:
            passes_filter = True
            if duration_constraint is not None:
                assessment_duration_raw = assessment.get('length_minutes')
                try:
                    if assessment_duration_raw is None:
                        passes_filter = False 
                        logger.debug(f"Filtering out '{assessment.get('name')}' (no duration, constraint is {duration_constraint})")
                    else:
                        assessment_duration = int(assessment_duration_raw)
                        if assessment_duration > duration_constraint:
                            passes_filter = False
                            logger.debug(f"Filtering out '{assessment.get('name')}' (duration {assessment_duration} > {duration_constraint})")
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse duration '{assessment_duration_raw}' for assessment '{assessment.get('name')}'. Filtering out.")
                    passes_filter = False 
            
            if passes_filter:
                pre_rerank_filtered_assessments.append(assessment)
        
        logger.info(f"After hard pre-rerank filters (e.g. max duration): {len(pre_rerank_filtered_assessments)} candidates remain.")

        if duration_constraint is not None and pre_rerank_filtered_assessments:
            logger.info(f"Adjusting candidate pool for duration preference: {duration_constraint} min.")
            candidates_after_duration_adj = self._adjust_for_duration(
                pre_rerank_filtered_assessments, 
                duration_constraint,
                bm25_candidate_pool_size 
            )
        else:
            candidates_after_duration_adj = pre_rerank_filtered_assessments[:bm25_candidate_pool_size]


        if not candidates_after_duration_adj:
            if assessments_for_filtering: 
                logger.warning("All candidates filtered out before LLM. Using top from BM25 pool (before hard filters) if available.")
                # Fallback to candidates before hard duration filter, but still apply soft duration sort and pool size limit
                if duration_constraint is not None:
                     candidates_for_llm = self._adjust_for_duration(assessments_for_filtering, duration_constraint, bm25_candidate_pool_size)
                else:
                     candidates_for_llm = assessments_for_filtering[:bm25_candidate_pool_size]
                if not candidates_for_llm: # If even this fallback yields nothing
                    logger.warning("Fallback to pre-filter pool also yielded no candidates.")
                    return []
            else: 
                logger.warning("No candidates found after BM25 filtering (empty initial pool).")
                return []
        else:
            candidates_for_llm = candidates_after_duration_adj
            
        if not candidates_for_llm:
            logger.warning("No candidates to send to LLM reranker after all filtering and adjustments.")
            if assessments_for_filtering:
                 logger.info(f"Final fallback: Returning top_k from initial BM25 pool of {len(assessments_for_filtering)} items (bypassing some filters).")
                 return assessments_for_filtering[:top_k]
            return [] 
            
        logger.info(f"Sending {len(candidates_for_llm)} candidates to LLM for reranking.")

        if not self.groq_client:
            logger.warning("Groq client not available. Skipping LLM reranking. Returning top results from processed candidate pool.")
            return candidates_for_llm[:top_k]
            
        return self._rerank_with_llm(criteria, candidates_for_llm, top_k)

    def _detect_cultural_focus(self, query: str) -> List[str]:
        """Identifies cultural context requirements from the query."""
        cultural_keywords_map = {
            'china': {'global','chinese', 'asia', 'mandarin', 'beijing', 'shanghai'},
            'india': {'indian', 'hindi', 'bangalore', 'mumbai', 'delhi'},
            'global': {'global', 'international', 'worldwide', 'cross-cultural', 'multinational'},
            'european': {'global','europe', 'european', 'eu', 'uk', 'german', 'french', 'spanish'}, # Example, can be more granular
            'american': {'global','american', 'us', 'usa', 'north america'},
            'collaboration': {'teamwork', 'business teams', 'stakeholders', 'cross-functional teams'},
            'diversity': {'diversity', 'inclusion', 'equity', 'belonging'},
        }
        query_lower = query.lower()
        detected_contexts = []
        for context_name, keywords in cultural_keywords_map.items():
            if any(word in query_lower for word in keywords): # Check if any keyword for a context is in the query
                # Check if the context_name itself is in the query (e.g. "china")
                # or if any of its specific keywords are in the query.
                # This logic is okay as `any(word in query_lower for word in keywords)` covers it.
                detected_contexts.append(context_name)
        if detected_contexts:
             logger.info(f"Detected cultural focus in query: {detected_contexts}")
        return list(set(detected_contexts)) 

    def _rerank_with_llm(self, criteria: Dict, 
                        candidates_for_reranking: List[Dict], top_n: int) -> List[Dict]:
        try:
            if not candidates_for_reranking:
                logger.warning("LLM Reranker: Received empty list of candidates. Returning empty.")
                return []

            candidate_details_list = []
            for i, assessment in enumerate(candidates_for_reranking):
                name = assessment.get('name', 'N/A')
                assessment_skills_list_raw = assessment.get('skills', [])
                # Ensure skills are strings for join and display
                assessment_skills_list = [str(s) for s in assessment_skills_list_raw if isinstance(s,str)]


                skills_display_count = 7
                skills_str = ", ".join(assessment_skills_list[:skills_display_count])
                if len(assessment_skills_list) > skills_display_count: skills_str += "..."
                
                duration = assessment.get('length_minutes', '?')
                
                assessment_domain_text_raw = assessment.get('domain', 'N/A')
                if isinstance(assessment_domain_text_raw, list): 
                    assessment_domain_text = ", ".join(map(str, assessment_domain_text_raw))
                else: 
                    assessment_domain_text = str(assessment_domain_text_raw)
                
                description_snippet = str(assessment.get('description', ''))
                if description_snippet and len(description_snippet) > 100:
                    description_snippet = description_snippet[:100].strip() + "..."
                
                details = (f"{i+1}. ID: {i+1} | Name: {name} | Domain: {assessment_domain_text} | "
                           f"Skills: {skills_str} | Duration: {duration} min | Desc: {description_snippet}")
                candidate_details_list.append(details)
            
            if not candidate_details_list: 
                logger.warning("No candidate details to send to LLM for reranking (list became empty). Returning original candidates.")
                return candidates_for_reranking[:top_n]
            candidate_list_str = "\n".join(candidate_details_list)
            
            requirements_str_parts = [
                f"- Primary Domain: \"{criteria['domain']}\"",
                f"- Key Skills: {', '.join(criteria['skills'])}"
            ]
            if criteria.get("duration_minutes") is not None:
                requirements_str_parts.append(f"- Target Maximum Duration: {criteria['duration_minutes']} minutes (This is a strong preference).")
            if criteria.get("experience_level") is not None:
                requirements_str_parts.append(f"- Target Experience Level: {criteria['experience_level']}.")
            if criteria.get("cultural_context"): 
                 requirements_str_parts.append(f"- Cultural Context Focus: {', '.join(criteria['cultural_context'])} (e.g., suitability for specific regions, global teams).")
            requirements_summary = "\n".join(requirements_str_parts)

            prompt = f"""You are an expert HR Tech consultant. Your task is to recommend the top {top_n} job assessments.
Evaluate each candidate assessment against the original job query and the extracted key requirements.

Original Job Query: "{criteria['raw_query']}"

Key Requirements Extracted:
{requirements_summary}

Available Assessments (IDs are 1-based from this list):
{candidate_list_str}

Instructions for your response:
1.  **Holistic Evaluation:** Evaluate each assessment against the original query AND all extracted key requirements (domain, skills, duration, experience, cultural context if specified).
2.  **Prioritization:**
    *   **Strong Match:** Prioritize assessments with a strong match to the primary domain and key skills.
    *   **Constraints & Preferences:** Pay close attention to constraints like maximum duration. Assessments exceeding specified limits should be significantly penalized or avoided unless they offer an overwhelmingly superior match in other critical areas. Also consider duration preferences if mentioned.
    *   **Cultural Context:** If a cultural context is specified, consider how well the assessment might align (e.g., language, regional norms if discernible, suitability for global roles).
    *   **Overall Relevance:** Consider the overall suitability of the assessment's content and scope for the job role implied by the query and requirements.
3.  **Output Format:** Return ONLY a valid JSON array of the assessment IDs (the number listed before 'Name', e.g., 'ID: 1' means you output 1) for your top {top_n} selections, in order of relevance (most relevant first). For example: [3, 1, 5].
    *   If you find fewer than {top_n} highly relevant assessments, return only those.
    *   If no assessments are suitably relevant, return an empty array [].
    *   Ensure the output is ONLY the JSON array, nothing else.
"""
            logger.debug(f"Reranking Prompt for LLM (first 1000 chars):\n{prompt[:1000]}...")
            response = self.groq_client.chat.completions.create(
                model=self.llm_model, 
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.05 
            )
            content = response.choices[0].message.content.strip()
            logger.debug(f"LLM Reranking Raw Response: {content}")
            
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            if not content: 
                logger.warning("LLM returned an empty string. Falling back.")
                return candidates_for_reranking[:top_n]

            selected_indices_from_llm_numbers = []
            try:
                parsed_data = json.loads(content)
                if isinstance(parsed_data, list):
                    selected_indices_from_llm_numbers = parsed_data
                elif isinstance(parsed_data, dict):
                    found_list = False
                    for val in parsed_data.values():
                        if isinstance(val, list):
                            selected_indices_from_llm_numbers = val
                            found_list = True
                            break
                    if not found_list:
                        logger.warning(f"LLM response was a dict but no list of IDs found. Content: {content}")
                else:
                    logger.warning(f"LLM response was not a list or a parsable dict containing a list. Content: {content}")

                reranked_assessments = []
                valid_llm_indices = [] 
                for id_val in selected_indices_from_llm_numbers:
                    try:
                        actual_idx = int(id_val) - 1 
                        if 0 <= actual_idx < len(candidates_for_reranking):
                            valid_llm_indices.append(actual_idx)
                        else:
                            logger.warning(f"LLM returned an out-of-bounds ID: {id_val} (1-based) for candidate pool size {len(candidates_for_reranking)}")
                    except (ValueError, TypeError):
                        logger.warning(f"LLM returned a non-integer ID value: {id_val}")
                
                added_names = set()
                for actual_idx in valid_llm_indices:
                    assessment = candidates_for_reranking[actual_idx]
                    if assessment['name'] not in added_names: 
                        reranked_assessments.append(assessment)
                        added_names.add(assessment['name'])
                
                if reranked_assessments:
                    logger.info(f"LLM reranked and selected {len(reranked_assessments)} assessments.")
                    if len(reranked_assessments) < top_n:
                        logger.info(f"LLM returned {len(reranked_assessments)} items, need {top_n}. Supplementing from the original candidate list sent to LLM.")
                        for cand in candidates_for_reranking: 
                            if len(reranked_assessments) >= top_n: break
                            if cand['name'] not in added_names: 
                                reranked_assessments.append(cand)
                                added_names.add(cand['name']) 
                    return reranked_assessments[:top_n]
                else:
                    logger.warning("LLM did not select any assessments or selection was invalid/empty. Falling back to top from pre-LLM candidate pool.")
                    return candidates_for_reranking[:top_n]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM reranking response JSON: {e}. Content: '{content}'. Falling back to top from pre-LLM candidate pool.")
                return candidates_for_reranking[:top_n]
        except Exception as e:
            logger.error(f"LLM reranking encountered an error: {e}. Falling back to top from pre-LLM candidate pool.")
            return candidates_for_reranking[:top_n] if candidates_for_reranking else []


# Main execution block for basic module testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHL Recommender System - Standalone Test")
    parser.add_argument("-q", "--query", type=str, help="A query string to get recommendations for.")
    parser.add_argument("-k", "--top_k", type=int, default=7, help="Number of top recommendations to return (default: 7).")
    args = parser.parse_args()

    logger.info("Initializing SHLRecommender for standalone test...")
    
    if not os.environ.get("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set. LLM features will be disabled for this test.")
    
    recommender = SHLRecommender()
    
    if not recommender.assessments:
        logger.error("No assessments were loaded. Standalone test cannot proceed effectively.")
    else:
        logger.info(f"SHLRecommender initialized with {len(recommender.assessments)} assessments.")
        
        query_to_process = args.query
        if not query_to_process:
            query_to_process = input("Please enter your query: ")

        if not query_to_process or not query_to_process.strip():
            logger.warning("No query provided. Exiting.")
        else:
            logger.info(f"\n--- Processing Query ---")
            logger.info(f"Query: {query_to_process}")
            
            recommendations = recommender.recommend(query_to_process, top_k=args.top_k)
            
            if recommendations:
                logger.info(f"Top {args.top_k} Recommendations for the query:")
                for r_idx, rec in enumerate(recommendations):
                    skills_display_raw = rec.get('skills', [])
                    if isinstance(skills_display_raw, list): 
                        skills_display = [str(s) for s in skills_display_raw if isinstance(s,str)][:3]
                    else: 
                        skills_display = 'N/A'
                    
                    logger.info(
                        f"  {r_idx+1}. Name: {rec.get('name')}, "
                        f"Domain: {rec.get('domain', 'N/A')}, "
                        f"Skills: {skills_display}, "
                        f"Duration: {rec.get('length_minutes', '?')} min"
                    )
            else:
                logger.info("No recommendations found for the query.")
        
    logger.info("\n--- Standalone test finished ---")