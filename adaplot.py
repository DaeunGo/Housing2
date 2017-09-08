from sklearn.metrics import mean_squared_error
dir(adaReg)
adaReg = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=300, learning_rate = 0.9)
adaReg.fit(X_train, y_train)
pred = adaReg.predict(X_test)
print("%.3f"%np.sqrt(mean_squared_error(y_test, pred)))
#plt.plot(adaReg.feature_importances_)
#adaReg.index

importances = adaReg.feature_importances_
indices = np.argsort(importances)[::-1]

indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), melbourne_df.columns) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()
