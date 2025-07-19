import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoints/checkpoint.pth'):
        """
        Args:
            patience (int):
            delta (float):
            path (str):
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_map = 0

    def __call__(self, val_map):
        score = val_map

        if self.best_score is None:
            self.best_score = score
            
            return True
            # self.save_checkpoint(model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(model)
            self.counter = 0

            return True

    def save_checkpoint(self, model, path = "checkpoints/checkpoint.pth"):
        """保存最佳模型"""
        torch.save(model.state_dict(), path)
        # print(f'Validation map decreased ({self.val_map_max:.6f} --> {val_map:.6f}). Saving model...')
        print(f'Validation map increase. Saving model...')

if __name__ == "__main__":
    # ========= 在训练循环中使用 =========
    early_stopping = EarlyStopping(patience=7, delta=0.001, path='best_model.pt')

    # for epoch in range(100):
    #     # 训练阶段
    #     model.train()
    #     train_loss = 0
    #     for batch in train_loader:
    #         # ... 正常训练步骤 ...
        
    #     # 验证阶段
    #     model.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             # ... 计算验证损失 ...
        
    #     val_loss /= len(val_loader)
        
    #     # 调用EarlyStopping
    #     early_stopping(val_loss, model)
        
    #     if early_stopping.early_stop:
    #         print("Early stopping triggered!")
    #         break

    # # 训练结束后加载最佳模型
    # model.load_state_dict(torch.load('best_model.pt'))