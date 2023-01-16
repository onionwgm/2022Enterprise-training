import os
import pickle as pickle
import numpy as np

import optim


class Solver(object):
    """
    训练器对象，使用方法如下：
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()

    求解器对传入的model的要求：
    model.params：一个dictionary，记录着模型相关的参数及系数，例如learning rate, reg, W, b, dW, db等
    model.loss(X, y)：能够在训练时计算出成本值和梯度，在测试时能够计算出score
      其中:
      X: (N, d_1, ..., d_k)
      y: (N,)
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个训练器对象
        必要输入参数：
        model
        data: 一个dictionary，包括4个key-value:
          'X_train': (N_train, d_1, ..., d_k)，训练特征矩阵
          'X_val': (N_val, d_1, ..., d_k)，验证特征矩阵
          'y_train': (N_train,)，训练数据的分类
          'y_val': (N_val,)，验证数据的分类
        可选输入参数：
        - update_rule: 指定一个optim.py文件中的函数名称，例如'sgd'。作为梯度更新策略
        - optim_config: 一个dictionary，包含有该模型的超参数设置。必须包含一个名为”learning_rate'的超参数
        - lr_decay: 每个epoch后，learning_rate的缩小比率
        - batch_size: mini-batch的每批次样本数量
        - num_epochs: 
        - print_every: 整数值。每隔print_every个iterations打印一次成本值
        - verbose: 布尔值。指示是否打印中间结果
        - num_train_samples: 训练样本数。如果不设置，则使用传入的所有训练样本
        - num_val_samples: 验证样本数。如果不设置，则使用传入的所有验证样本
        - checkpoint_name: 如果不为None，则每个epoch后将结果保存到名为checkpoint_name开头的文件中
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # 获取其余key-value对设定的附加参数
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # 附加参数到此为止。如果还有其它附加参数，则抛出一个错误
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # 确保"update_rule"附加参数确实传入了一个值
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        # 设置self.update_rule指向optim的同名函数(默认是sgd)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        """
        初始化变量、集合等以保存计算过程中的重要输出值
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # 将model中可能自带的param和self.optim_config中的param一起合并到self.optim_configs集合中
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        执行一次iteration
        """
        # 获取mini-batch数据集
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算成本值和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 执行梯度更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _save_checkpoint(self):
        # 保存
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        计算正确率
        """

        # 如果有设置，仅对部分样本统计正确率
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 对每个batch的数据分别进行预测
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        """
        执行训练
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # 打印成本值
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # 在每个epoch之后，下调learning rate
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            # 在第一个iteration、最后一个iteration以及每个epoch结束后
            # 计算训练和验证数据正确率
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                    num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # 记录性能最好的模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # 训练结束后，将最佳的模型参数存放回模型中
        self.model.params = self.best_params
