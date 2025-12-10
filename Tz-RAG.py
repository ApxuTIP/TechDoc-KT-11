import json
import numpy as np
from typing import List, Dict, Any

class SimpleRAGSystem:
    def __init__(self, vectors_file='tz_vectors.json'):
        """Инициализация простой RAG системы"""
        with open(vectors_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vectors = data['vectors']
            
        self.embeddings_matrix = np.array([vec['embedding'] for vec in self.vectors])
        print(f"✅ Загружено {len(self.vectors)} блоков из ТЗ")
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """Простая векторизация запроса через TF-IDF like подход"""

        all_texts = [vec['text'] for vec in self.vectors]
        all_words = set()
        for text in all_texts:
            words = text.lower().split()
            all_words.update(words)
        
        query_words = query.lower().split()
        query_vector = np.zeros(len(self.vectors[0]['embedding']))
        
        for i, word in enumerate(query_words):
            if i < len(query_vector):
                query_vector[i] = 1.0 / (i + 1)  # Уменьшаем вес для последующих слов
      
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
            
        return query_vector
    
    def find_similar_vectors(self, query: str, top_k: int = 3) -> List[Dict]:
        """Находит наиболее похожие векторы на запрос"""

        query_vector = self.vectorize_query(query)
        
        if query_vector is None:
            return []
        
        similarities = np.dot(self.embeddings_matrix, query_vector)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            vec = self.vectors[idx]
            similarity = float(similarities[idx])
            
            results.append({
                'id': vec['id'],
                'section': vec['section'],
                'text': vec['text'],
                'hash': vec['hash'],
                'similarity': similarity,
                'relevance_percentage': min(100, max(0, (similarity + 1) * 50))
            })
        
        results.sort(key=lambda x: x['relevance_percentage'], reverse=True)
        return results
    
    def generate_response(self, query: str, context_blocks: List[Dict]) -> Dict[str, Any]:
        """Генерирует ответ на основе найденных блоков"""
        if not context_blocks:
            return {
                'query': query,
                'response': "По вашему запросу не найдено релевантной информации в ТЗ.",
                'suggestion': "Попробуйте использовать ключевые слова из ТЗ: процедурная генерация, кодекс, RPG, планеты, MVP и т.д."
            }
        
        response_parts = []
        response_parts.append(f"🔍 **Ответ на запрос:** '{query}'")
        response_parts.append("")
        response_parts.append("📋 **Найденная информация в ТЗ:**")
        
        for i, block in enumerate(context_blocks[:3]):
            response_parts.append(f"\n{i+1}. **{block['section']}** (релевантность: {block['relevance_percentage']:.1f}%)")
            
            lines = block['text'].split('\n')
            preview = ' | '.join([line.strip() for line in lines if line.strip()][:2])
            if len(preview) > 150:
                preview = preview[:147] + "..."
            response_parts.append(f"   {preview}")
        
        response_parts.append("\n📝 **Ключевые моменты:**")
        
        keywords = set()
        for block in context_blocks[:3]:
          
            lines = block['text'].split('\n')
            for line in lines:
                words = line.split(':')
                if len(words) > 1:
                    keywords.add(words[0].strip())
        
        for keyword in list(keywords)[:5]:
            response_parts.append(f"   • {keyword}")
        
        response_parts.append(f"\n📊 **Статистика:** Найдено {len(context_blocks)} релевантных блоков, средняя релевантность: {np.mean([b['relevance_percentage'] for b in context_blocks]):.1f}%")
        
        response = "\n".join(response_parts)
        
        return {
            'query': query,
            'context_blocks': context_blocks,
            'response': response,
            'total_blocks_found': len(context_blocks),
            'avg_relevance': np.mean([b['relevance_percentage'] for b in context_blocks])
        }
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Основной метод для выполнения запроса"""
        print(f"\n{'='*60}")
        print(f"🎯 Запрос: {question}")
        print('='*60)
        
        similar_blocks = self.find_similar_vectors(question, top_k)
        
        relevant_blocks = [b for b in similar_blocks if b['relevance_percentage'] > 30]
        
        response = self.generate_response(question, relevant_blocks)
        
        print("\n" + response['response'])
        
        print(f"\n{'='*60}")
        print("🔧 Техническая информация:")
        print(f"   • Найдено блоков: {response['total_blocks_found']}")
        print(f"   • Средняя релевантность: {response['avg_relevance']:.1f}%")
        
        if response['context_blocks']:
            print(f"\n📚 Использованные блоки ТЗ:")
            for block in response['context_blocks'][:3]:
                print(f"   [{block['hash']}] {block['section'][:40]}... ({block['relevance_percentage']:.1f}%)")
        
        print('='*60)
        
        return response

def main():
    """Главная функция для тестирования системы"""
    
    print("🚀 Загрузка RAG системы для ТЗ 'Звездный Кодекс'...")
    rag_system = SimpleRAGSystem('tz_vectors.json')
    
    test_queries = [
        "Какие образовательные дисциплины есть в игре?",
        "Как работает процедурная генерация?",
        "Какие платформы поддерживает игра?",
        "Что такое MVP?",
        "Какие навыки у персонажа?",
        "Как работает система сохранения?",
        "Какие форматы данных используются?",
        "Что такое Кодекс в игре?",
        "Какие угрозы есть в игре?",
        "Какой стиль графики?"
    ]
    
    print("\n🧪 Запуск тестовых запросов...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*60}")
        print(f"Тест {i}/{len(test_queries)}")
        rag_system.query(query)
        
        if i < len(test_queries):
            input("\nНажмите Enter для следующего запроса...")
    
    print(f"\n{'='*60}")
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*60)
    print("Вводите запросы на русском языке.")
    print("Примеры хороших запросов:")
    print("  - 'образование в игре'")
    print("  - 'генерация планет'")
    print("  - 'игровая механика'")
    print("  - 'арт и звук'")
    print("Введите 'выход' для завершения.\n")
    
    while True:
        user_query = input("🎯 Ваш запрос: ").strip()
        
        if user_query.lower() in ['выход', 'exit', 'quit', 'q']:
            print("Завершение работы...")
            break
        
        if not user_query:
            continue
        
        try:
            rag_system.query(user_query)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("Попробуйте другой запрос.")

if __name__ == "__main__":
    main()
