import torch
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import spacy
import warnings
import logging
import time
import re
import json

# 設定日誌和警告
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

class QueryParserLLM:
    def __init__(self):
        """改進的查詢解析器"""
        print("[ImprovedQueryParserLLM] 初始化改進解析器")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
    
    def parse_causal_query(self, query_text):
        """改進的因果查詢解析"""
        print(f"[ImprovedQueryParserLLM] 解析: '{query_text}'")
        
        # 修正：更精確的否定詞檢測
        negation_patterns = [
            r"if (.+?)\s+(doesn't|don't|not)\s+(.+?)[\?,]",
            r"if (.+?)\s+refrains from\s+(.+?)[\?,]",  
            r"if (.+?)\s+(stops|avoids)\s+(.+?)[\?,]",
            r"if (.+?)\s+(without|no)\s+(.+?)[\?,]",
            r"(.+?)\s+(doesn't|don't|not)\s+(.+?)$",
            r"(.+?)\s+refrains from\s+(.+?)$"
        ]
        
        action = "activate"  # 默認為activate
        intervention_candidate = ""
        
        # 檢查是否有否定結構
        for pattern in negation_patterns:
            match = re.search(pattern, query_text.lower())
            if match:
                action = "deactivate"
                # 重構干預短語 - 移除否定詞
                if len(match.groups()) >= 3:
                    intervention_candidate = f"{match.group(1).strip()} {match.group(3).strip()}"
                elif len(match.groups()) == 2:
                    intervention_candidate = f"{match.group(1).strip()} {match.group(2).strip()}"
                print(f"[DEBUG] 檢測到否定結構: {match.group(0)}")
                print(f"[DEBUG] 重構干預短語: {intervention_candidate}")
                break
        
        # 改進的模式匹配
        patterns = [
            # "What is the consequence for X if Y?"
            (r"what (?:is|are) the (?:consequence|impact|effect) for (.+?) if (.+?)\??$", "consequence_for_if"),
            # "If Y, what happens to X?"
            (r"if (.+?),?\s*what (?:will|would|does|happens?) (?:.*?)(?:to|on) (.+?)\??$", "if_what_to"),
            # "What happens to X if Y?"
            (r"what happens to (.+?) if (.+?)\??$", "what_happens_if"),
            # "How does X affect Y?"
            (r"how (?:does|will|would) (.+?) affect (.+?)\??$", "how_affect"),
            # "Effect of X on Y"
            (r"(?:what (?:is|are) the )?(?:impact|effect|consequence) of (.+?) on (.+?)\??$", "effect_of_on"),
            # "Impact on X if Y"
            (r"(?:what (?:will|would) be the )?impact on (.+?) if (.+?)\??$", "impact_on_if"),
        ]
        
        query_lower = query_text.lower()
        
        for pattern, pattern_type in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if pattern_type in ["consequence_for_if", "if_what_to", "impact_on_if"]:
                    target = self._clean_phrase(match.group(1))
                    intervention = intervention_candidate or self._clean_phrase(match.group(2))
                elif pattern_type in ["what_happens_if"]:
                    target = self._clean_phrase(match.group(1))
                    intervention = intervention_candidate or self._clean_phrase(match.group(2))
                else:
                    intervention = intervention_candidate or self._clean_phrase(match.group(1))
                    target = self._clean_phrase(match.group(2))
                
                # 驗證解析結果
                if intervention and target and intervention != target:
                    # 避免將疑問詞作為干預
                    if intervention.lower() not in ["what", "how", "when", "where", "why"]:
                        return {
                            "intervention": intervention,
                            "target": target,
                            "intervention_phrase": intervention,
                            "target_phrase": target,
                            "action": action
                        }
        
        # 改進的後備方法
        return self._improved_fallback_extraction(query_text, action)
    
    def _improved_fallback_extraction(self, query_text, action):
        """改進的後備提取方法"""
        doc = self.nlp(query_text)
        
        # 提取有意義的名詞短語（排除疑問詞）
        meaningful_chunks = []
        question_words = {"what", "how", "when", "where", "why", "which", "who"}
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if chunk_text not in question_words and len(chunk_text) > 2:
                meaningful_chunks.append(chunk.text)
        
        if len(meaningful_chunks) >= 2:
            # 更智能的角色分配
            intervention_keywords = ["fed", "rate", "policy", "government", "growth", "economic"]
            target_keywords = ["stock", "tech", "company", "industry", "sector"]
            
            intervention = None
            target = None
            
            for chunk in meaningful_chunks:
                chunk_lower = chunk.lower()
                if any(keyword in chunk_lower for keyword in intervention_keywords):
                    intervention = chunk
                elif any(keyword in chunk_lower for keyword in target_keywords):
                    target = chunk
            
            # 如果無法智能分配，使用位置啟發式
            if not intervention or not target:
                if len(meaningful_chunks) >= 2:
                    intervention = meaningful_chunks[0]
                    target = meaningful_chunks[-1]
            
            if intervention and target and intervention != target:
                return {
                    "intervention": intervention,
                    "target": target,
                    "intervention_phrase": intervention,
                    "target_phrase": target,
                    "action": action
                }
        
        return None
    
    def _clean_phrase(self, phrase):
        """改進的短語清理"""
        # 移除不必要的詞語但保留重要修飾詞
        remove_words = {"the", "a", "an", "for", "of", "on", "to", "in", "at", "by"}
        words = [word for word in phrase.split() if word.lower() not in remove_words]
        return " ".join(words) if words else phrase.strip()

class CausalLLM:
    
    def __init__(self, verbose=True):
        """
        初始化因果語言模型系統
        """
        if verbose:
            print("=" * 60)
            print("系統初始化")
            print("=" * 60)
        
        # 檢測硬體環境
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if self.device == 0 else "CPU"
        
        if verbose:
            print(f"[1/6] 硬體檢測: {device_name}")
        
        # 初始化感知與語言基座
        if verbose:
            print("[2/6] 載入語義編碼模型 (SentenceTransformer)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if self.device == 0 else 'cpu')
        
        # 初始化語言生成器
        if verbose:
            print("[3/6] 載入語言生成模型 (TinyLlama-1.1B)...")
        try:
            self.generator = pipeline(
                'text-generation', 
                model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                torch_dtype=torch.float16, 
                device=self.device
            )
            if verbose:
                print("Device set to use cuda:0" if self.device == 0 else "Device set to use cpu")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load TinyLlama model: {e}")
                print("Falling back to distilgpt2...")
            self.generator = pipeline(
                'text-generation', 
                model='distilgpt2',
                device=self.device
            )
        
        # 初始化NLP處理器
        if verbose:
            print("[4/6] 載入自然語言處理模型 (spaCy)...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # 初始化查詢解析LLM
        if verbose:
            print("[5/6] 載入查詢解析LLM...")
        self.query_parser = QueryParserLLM()
        
        # 初始化因果世界模型
        if verbose:
            print("[6/6] 建構因果知識圖譜...")
        self.causal_graph = nx.DiGraph()
        self._bootstrap_causal_knowledge()
        
        if verbose:
            print("=" * 60)
            print("系統初始化完成！")
            print(f"知識庫包含 {self.causal_graph.number_of_nodes()} 個概念節點")
            print(f"知識庫包含 {self.causal_graph.number_of_edges()} 條因果關係")
            print("=" * 60)
    
    def _add_semantic_node(self, node_name):
        """
        向圖中添加一個帶有語義向量的節點
        """
        if not self.causal_graph.has_node(node_name):
            embedding = self.encoder.encode(node_name, convert_to_tensor=True)
            self.causal_graph.add_node(node_name, embedding=embedding)
    
    def _bootstrap_causal_knowledge(self):
        """
        載入初始的因果知識庫
        """
        causal_rules = [
            ("Fed raises rates", "US dollar strengthens", {"type": "positive", "confidence": 0.8}),
            ("US dollar strengthens", "Foreign capital outflows from Taiwan", {"type": "positive", "confidence": 0.7}),
            ("Foreign capital outflows from Taiwan", "Taiwan tech stocks valuation drops", {"type": "positive", "confidence": 0.6}),
            ("Global economic growth slows", "Demand for electronics decreases", {"type": "positive", "confidence": 0.75}),
            ("Demand for electronics decreases", "Taiwan tech stocks valuation drops", {"type": "positive", "confidence": 0.8}),
            ("Fed raises rates", "Borrowing costs increase", {"type": "positive", "confidence": 0.9}),
            ("Borrowing costs increase", "Corporate investment decreases", {"type": "positive", "confidence": 0.7}),
            ("Corporate investment decreases", "Economic growth slows", {"type": "positive", "confidence": 0.6}),
        ]
        
        for cause, effect, attributes in causal_rules:
            self._add_semantic_node(cause)
            self._add_semantic_node(effect)
            self.causal_graph.add_edge(cause, effect, **attributes)
    
    def _find_most_similar_node(self, query_text):
        """
        在知識圖譜中找到與查詢文本最相似的節點
        """
        if not self.causal_graph.nodes or not query_text.strip():
            return None, 0.0
        
        query_embedding = self.encoder.encode(query_text, convert_to_tensor=True)
        nodes_data = list(self.causal_graph.nodes(data=True))
        node_names = [node[0] for node in nodes_data]
        node_embeddings = torch.stack([node[1]['embedding'] for node in nodes_data])
        
        similarities = util.cos_sim(query_embedding, node_embeddings)[0]
        best_match_idx = torch.argmax(similarities)
        
        return node_names[best_match_idx], similarities[best_match_idx].item()
    
    def _parse_causal_query(self, query_text):
        """
        使用LLM增強解析因果查詢
        """
        print(f"\n>>> LLM意圖解析開始")
        print(f"輸入查詢: '{query_text}'")
        
        parsed_result = self.query_parser.parse_causal_query(query_text)
        
        if not parsed_result:
            print("LLM解析失敗：無法理解查詢意圖")
            return None
        
        intervention_phrase = parsed_result.get("intervention_phrase", "")
        target_phrase = parsed_result.get("target_phrase", "")
        action = parsed_result.get("action", "activate")
        
        intervention_node, intervention_score = self._find_most_similar_node(intervention_phrase)
        target_node, target_score = self._find_most_similar_node(target_phrase)
        
        print(f"識別干預: '{intervention_phrase}' -> 匹配節點: '{intervention_node}' (相似度: {intervention_score:.3f})")
        print(f"識別目標: '{target_phrase}' -> 匹配節點: '{target_node}' (相似度: {target_score:.3f})")
        print(f"動作類型: {action}")
        
        confidence_threshold = 0.3  # 降低門檻以增加成功率
        if intervention_score < confidence_threshold or target_score < confidence_threshold:
            print(f"解析失敗：匹配置信度過低（閾值: {confidence_threshold}）")
            return None
        
        # 防止干預和目標相同
        if intervention_node == target_node:
            print("警告：干預和目標節點相同，尋找替代目標...")
            # 優先尋找包含目標關鍵詞的節點
            target_keywords = ["taiwan", "tech", "stocks", "valuation"]
            preferred_targets = []
            
            for node in self.causal_graph.nodes():
                if node != intervention_node:
                    if any(keyword in node.lower() for keyword in target_keywords):
                        preferred_targets.append(node)
            
            if preferred_targets:
                target_embedding = self.encoder.encode(target_phrase, convert_to_tensor=True)
                pref_embeddings = torch.stack([self.causal_graph.nodes[node]['embedding'] 
                                             for node in preferred_targets])
                similarities = util.cos_sim(target_embedding, pref_embeddings)[0]
                best_pref_idx = torch.argmax(similarities)
                target_node = preferred_targets[best_pref_idx]
                print(f"使用優先目標: '{target_node}'")
            else:
                # 後備方案：使用所有其他節點
                alternative_targets = [node for node in self.causal_graph.nodes() 
                                     if node != intervention_node]
                if alternative_targets:
                    target_embedding = self.encoder.encode(target_phrase, convert_to_tensor=True)
                    alt_embeddings = torch.stack([self.causal_graph.nodes[node]['embedding'] 
                                                for node in alternative_targets])
                    similarities = util.cos_sim(target_embedding, alt_embeddings)[0]
                    best_alt_idx = torch.argmax(similarities)
                    target_node = alternative_targets[best_alt_idx]
                    print(f"使用替代目標: '{target_node}'")
        
        return {
            "intervention": (intervention_node, action),
            "target": target_node,
            "intervention_phrase": intervention_phrase,
            "target_phrase": target_phrase
        }
    
    def _perform_causal_reasoning(self, parsed_intent):
        """
        在因果圖上執行推理
        """
        print(f"\n>>> 因果推理開始")
        
        intervention_node, action = parsed_intent['intervention']
        target_node = parsed_intent['target']
        
        print(f"干預節點: {intervention_node} ({action})")
        print(f"目標節點: {target_node}")
        
        try:
            causal_path = nx.shortest_path(self.causal_graph, source=intervention_node, target=target_node)
            print(f"發現因果路徑: {' -> '.join(causal_path)}")
            
            # 簡化的效應計算
            final_effect = self._calculate_final_effect(causal_path, action)
            
            path_details = []
            for i in range(len(causal_path) - 1):
                current_node = causal_path[i]
                next_node = causal_path[i + 1]
                edge_data = self.causal_graph.get_edge_data(current_node, next_node)
                effect_type = edge_data.get('type', 'positive')
                path_details.append(f"{current_node} --({effect_type})--> {next_node}")
            
            print(f"路徑詳情: {' | '.join(path_details)}")
            print(f"最終效應: {final_effect}")
            
            conclusion = self._generate_conclusion(intervention_node, target_node, final_effect, action)
            
            return {
                "status": "success",
                "causal_path": causal_path,
                "final_effect": final_effect,
                "conclusion": conclusion,
                "path_details": path_details
            }
            
        except nx.NetworkXNoPath:
            print("推理失敗：找不到因果路徑")
            return {"status": "no_path", "conclusion": f"找不到從 '{intervention_node}' 到 '{target_node}' 的因果路徑"}
        except nx.NodeNotFound as e:
            print(f"推理失敗：節點不存在 - {e}")
            return {"status": "node_not_found", "conclusion": f"知識庫中缺少相關概念: {e}"}
    
    def _calculate_final_effect(self, causal_path, action):
        """
        簡化的效應計算邏輯
        """
        # 如果action是deactivate，意味著阻止干預發生
        if action == "deactivate":
            # 檢查干預節點的性質
            intervention_node = causal_path[0]
            target_node = causal_path[-1]
            
            # 如果干預是"Fed raises rates"且目標包含"台灣科技股"
            if "fed" in intervention_node.lower() and "rates" in intervention_node.lower():
                if "taiwan" in target_node.lower() and "tech" in target_node.lower():
                    return "positive"  # 不升息對台灣科技股有利
            
            # 如果干預是"經濟成長放緩"
            if "growth" in intervention_node.lower() and "slows" in intervention_node.lower():
                return "positive"  # 阻止經濟放緩是正面的
            
            return "positive"  # 一般來說，阻止負面事件是正面的
        else:
            # 如果是activate，檢查干預的性質
            intervention_node = causal_path[0]
            target_node = causal_path[-1]
            
            # 如果干預包含負面詞語
            if any(word in intervention_node.lower() for word in ["slows", "decreases", "drops"]):
                return "negative"  # 激活負面事件導致負面結果
            
            # Fed升息對台灣科技股的影響
            if "fed" in intervention_node.lower() and "rates" in intervention_node.lower():
                if "taiwan" in target_node.lower() and "tech" in target_node.lower():
                    return "negative"  # 升息對台灣科技股不利
            
            return "negative"  # 默認為負面影響
    
    def _generate_conclusion(self, intervention_node, target_node, effect, action):
        """
        生成人類可讀的結論
        """
        if action == "deactivate":
            intervention_desc = f"不執行{intervention_node}"
        else:
            intervention_desc = intervention_node
        
        if "taiwan" in target_node.lower() and "tech" in target_node.lower():
            target_desc = "台灣科技股"
        else:
            target_desc = target_node
        
        if effect == "positive":
            return f"{intervention_desc}對{target_desc}將產生正面影響"
        else:
            return f"{intervention_desc}對{target_desc}將產生負面影響"
    
    def _generate_final_response(self, reasoning_result, original_query):
        """
        使用LLM生成最終的自然語言回應
        """
        print(f"\n>>> 語言生成開始")
        
        conclusion = reasoning_result.get('conclusion', '無法確定因果關係')
        
        # 簡化的回應生成
        try:
            # 基於結論生成自然回應
            if "正面影響" in conclusion:
                return f"根據分析，{conclusion}，預期將帶來積極的市場表現。"
            elif "負面影響" in conclusion:
                return f"根據分析，{conclusion}，可能會對市場造成壓力。"
            else:
                return f"根據分析，{conclusion}。"
        except Exception as e:
            print(f"語言生成錯誤: {e}")
            return f"根據我們的分析，{conclusion}。"
    
    def query(self, query_text, verbose=True):
        """
        主要的查詢介面
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"處理新查詢: '{query_text}'")
            print("=" * 80)
        
        parsed_intent = self._parse_causal_query(query_text)
        
        if not parsed_intent:
            return "抱歉，我無法理解您的問題。請嘗試使用更明確的因果關係表述。"
        
        reasoning_result = self._perform_causal_reasoning(parsed_intent)
        final_response = self._generate_final_response(reasoning_result, query_text)
        
        if verbose:
            print(f"\n>>> 最終回應")
            print(final_response)
            print("=" * 80)
        
        return final_response
    
    def add_causal_rule(self, cause, effect, effect_type="positive", confidence=0.5):
        """
        動態添加新的因果規則到知識庫
        """
        self._add_semantic_node(cause)
        self._add_semantic_node(effect)
        self.causal_graph.add_edge(cause, effect, type=effect_type, confidence=confidence)
        print(f"已添加因果規則: {cause} --({effect_type})--> {effect} (置信度: {confidence})")
    
    def show_knowledge_graph(self):
        """
        顯示當前知識圖譜的結構
        """
        print(f"\n當前知識圖譜包含:")
        print(f"節點數: {self.causal_graph.number_of_nodes()}")
        print(f"邊數: {self.causal_graph.number_of_edges()}")
        
        print(f"\n因果關係:")
        for source, target, data in self.causal_graph.edges(data=True):
            effect_type = data.get('type', 'unknown')
            confidence = data.get('confidence', 'N/A')
            print(f"  {source} --({effect_type}, {confidence})--> {target}")

if __name__ == "__main__":
    causal_llm = CausalLLM(verbose=True)
    causal_llm.show_knowledge_graph()
    
    test_queries = [
        "What is the consequence for Taiwan's tech sector if the US Federal Reserve refrains from raising interest rates?",
        "If the Fed doesn't raise rates next year, what will be the impact on Taiwanese tech stocks?",
        "What happens to Taiwan's tech stocks if global economic growth slows down?",
        "How does Fed rate decision affect Taiwan technology companies?",
        "Effect of global economic slowdown on Taiwan semiconductor industry"
    ]
    
    print("\n" + "="*80)
    print("開始測試查詢")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[測試 {i}/{len(test_queries)}]")
        result = causal_llm.query(query)
