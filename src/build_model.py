from train_single import SingleModel

single = SingleModel()
x_train, y_train, x_val, y_val, x_test, y_test = single.load_data(
    './datasets/full_single_mix2.csv')
model = single.create_model()

if __name__ == '__main__':
    single.train_evaluate_model(model, x_train, y_train,
                                x_val, y_val, x_test, y_test)
