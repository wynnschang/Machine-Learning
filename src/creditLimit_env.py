import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


class CreditLimitEnv:
    def __init__(self, df, provision_bins):
        """
               Expected columns in `df`:
               - BALANCE_CLASS       : Customer behavioral class (0 = light, 1 = heavy)
               - UR_456              : Utilization rate
               - PR_456              : Payment rate
               - UR_789              : Utilization rate
               - PR_789              : Payment rate
               - INT                 : Interest rate
               - L_P                 : Current limit
               - L_P_789             : Future limit proxy
               - AVG_BALANCE_456     : Average balance over window 456
               - AVG_BALANCE_789     : Future balance proxy
               - AVG_BALANCE_123     : Ground truth future balance
               - L_R                 : Reference limit (baseline for ΔPROVISION)

               Note:
               - 'D_PROVISION_bin' will be computed dynamically during environment steps.
               - 'BAL_3_pred' is created during environment initialization (via regression model).
               """
        self.df = df[df['BALANCE_CLASS'].isin([0, 1])].reset_index(drop=True)
        self.provision_bins = provision_bins

        # self.state_space = ['BALANCE_CLASS', 'UR_456', 'PR_456', 'delta_prov'] # continuous D_provision
        self.state_space = ['BALANCE_CLASS', 'UR_456', 'PR_456', 'D_PROVISION_bin'] # discrete D_provision
        self.action_space = [0, 1]
        self.n_customers = len(self.df)
        self.current_step = 0

        self.beta = 0.2
        self.pd_dict = {0: 0.01, 1: 0.05, 2: 0.15}
        self.lgd = 0.5
        self.ccf = 0.8

        self.model_0 = None
        self.model_1 = None

        self.train_predictor_model()
        self.predict_bal_3_from_456()

        self.df['D_PROVISION_bin'] = 0
        # self.df['delta_prov'] = 0.0

        self.state_matrix = self.df[self.state_space].values.copy()

    def train_predictor_model(self):
        features_789 = ['UR_789', 'PR_789', 'INT', 'L_P_789', 'AVG_BALANCE_789']
        target = 'AVG_BALANCE_456'

        model_grid = {
            'rf': {
                'model': RandomForestRegressor(random_state=42),
                'params': {'n_estimators': [100], 'max_depth': [10], 'max_features': [0.8]}
            },
            'gbr': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
            }
        }

        for cls in [0, 1]:
            subset = self.df[self.df['BALANCE_CLASS'] == cls]
            if subset.empty:
                continue

            X = subset[features_789]
            y = subset[target]
            best_score, best_model = -np.inf, None

            for cfg in model_grid.values():
                grid = GridSearchCV(cfg['model'], cfg['params'], cv=2, scoring='r2', n_jobs=-1)
                grid.fit(X, y)
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_model = grid.best_estimator_

            if cls == 0:
                self.model_0 = best_model
            else:
                self.model_1 = best_model

    def predict_bal_3_from_456(self):
        # Define mapping from 456 to 789 naming
        feature_map = {
            'UR_456': 'UR_789',
            'PR_456': 'PR_789',
            'INT': 'INT',
            'L_P': 'L_P_789',
            'AVG_BALANCE_456': 'AVG_BALANCE_789'
        }

        # Must match this exact order
        features_ordered = ['UR_789', 'PR_789', 'INT', 'L_P_789', 'AVG_BALANCE_789']
        # Rename and reorder
        df_renamed = self.df[list(feature_map.keys())].rename(columns=feature_map)
        df_renamed = df_renamed[features_ordered]

        preds = np.zeros(len(self.df))

        for cls, model in [(0, self.model_0), (1, self.model_1)]:
            if model is None:
                continue
            mask = self.df['BALANCE_CLASS'] == cls
            preds[mask] = model.predict(df_renamed.loc[mask])

        self.df['BAL_3_pred'] = preds

    def reset(self):
        self.current_step = np.random.randint(0, self.n_customers)
        return self.state_matrix[self.current_step]

    def step(self, action):
        row = self.df.iloc[self.current_step]
        new_l_p = row['L_P'] * (1 + self.beta) if action == 1 else row['L_P']
        bal_3_pred = row['BAL_3_pred']
        pd_value = self.pd_dict[row['BALANCE_CLASS']]

        reward = (
            3 * row['INT'] * row['AVG_BALANCE_456'] * (1 - pd_value)
            - pd_value * self.lgd * (bal_3_pred + self.ccf * (new_l_p - bal_3_pred))
        )
        reward = np.clip(reward, -1e6, 1e6) / 1e6

        delta_prov = (new_l_p - row['L_R']) / row['L_R'] if row['L_R'] != 0 else 0
        # for discrete D_provision
        new_dp_bin = pd.cut([delta_prov], bins=self.provision_bins, labels=False, include_lowest=True)
        new_dp_bin = int(new_dp_bin[0]) if new_dp_bin.size > 0 and not pd.isna(new_dp_bin[0]) else 0

        new_state = self.state_matrix[self.current_step].copy()
        new_state[self.state_space.index('D_PROVISION_bin')] = new_dp_bin
        # for continuous D_provision
        # new_state = self.state_matrix[self.current_step].copy()
        # new_state[self.state_space.index('delta_prov')] = delta_prov

        info = {
            'BALANCE_CLASS': row['BALANCE_CLASS'],
            'actual_balance': row['AVG_BALANCE_123'],
            'interest_rate': row['INT'],
            'pd': pd_value,
            'lgd': self.lgd,
            'new_limit': new_l_p,
            'ccf': self.ccf,
            'delta_provision': delta_prov
        }

        if os.environ.get("DEBUG", "0") == "1":
            print(f"\n[ENV DEBUG] Step Index: {self.current_step}")
            print(f"Action: {action}, New Limit: {new_l_p:.2f}, PD: {pd_value}")
            print(f"BAL_3 Pred: {bal_3_pred:.2f}, Reward: {reward:.4f}")
            print(f"Delta Prov Bin: {new_dp_bin}, GT Balance: {info['actual_balance']:.2f}") # discrete prov
            # print(f"Delta Prov: {delta_prov:.4f}, GT Balance: {info['actual_balance']:.2f}") # continuous prov
            print(f"BALANCE_CLASS: {row['BALANCE_CLASS']}")

        self.current_step = (self.current_step + 1) % self.n_customers
        done = self.current_step == 0
        return new_state, reward, done, info