class UserDatabaseRouter:
    '''
    특정 사용자의 모든 데이터 베이스 기능을 컨트롤 하는 라우터
    '''

    def db_for_read(self, model, **hints):
        # 현재 로그인한 사용자의 DB로 라우팅
        user = hints.get('user')
        if user:
            return f'user_{user.username}_db'
        return 'default'
    
    def db_for_write(self, model, **hints):
        # 현재 로그인한 사용자의 DB로 라우팅
        user = hints.get('user')
        if user:
            return f'user_{user.username}_db'
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        # 같은 DB 안의 관계만 허용
        if hasattr(obj1, 'user') and hasattr(obj2, 'user'):
            if obj1.user == obj2.user:
                return True
        return None
    
    def allow_migrate(self, db, app_label, model_name = None, **hints):
        # 기본 DB에 마이그레이션 허용 (auth나 session 같은 공통 테이블)
        if db == 'default':
            return True
        return False


