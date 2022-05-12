import torch.nn as nn
import torch.nn.functional as fc
import torch.nn.init as init


class fc_model(nn.Module):

    def __init__(self, input_size, num_classes=11, dropout=0.5):
        super(fc_model, self).__init__()

        # cnn model
        # first conv layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.nor = nn.BatchNorm2d(64, affine=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 112*112

        # second conv layer
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 56*56
        # self.nor3 = nn.BatchNorm2d(128, affine=True)

        # # third conv layer
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        # self.nor4 = nn.BatchNorm2d(256, affine=True)
        # self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 28*28

        # forth conv layer
        # self.conv4_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        # self.nor4_5 = nn.BatchNorm2d(256, affine=True)

        # fifth conv layer
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        # self.nor5 = nn.BatchNorm2d(512, affine=True)
        # self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 14*14

        # # sixth conv layer
        # self.conv5_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        # self.nor5_6 = nn.BatchNorm2d(512, affine=True)

        # # seventh conv layer
        # self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        # self.nor6 = nn.BatchNorm2d(512, affine=True)

        # eighth conv layer
        # self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=(1, 1))
        # self.nor7 = nn.BatchNorm2d(1024, affine=True)
        #
        # # ninth conv layer
        # self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=(1, 1))
        # self.nor8 = nn.BatchNorm2d(1024, affine=True)
        # self.pool8 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 7*7

        # self.dropout= nn.Dropout(0.5)

        # 7 pooling layer
        # self.linear = nn.Linear(7 *7 * 1024, 2048)
        #
        # self.linear3 = nn.Linear(2048, 1024)
        #
        # self.linear4 = nn.Linear(1024, num_classes)
        self.linear = nn.Linear(112 * 112 * 64, 64)

        # self.linear3 = nn.Linear(2048, 1024)

        self.linear4 = nn.Linear(64, num_classes)
        # init the weights for conv, linear layer, and batch normalization layer
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, image_vectorized):
        """
            feed-forward (vectorized) image into a linear model for classification.
        """
        image_vectorized = fc.relu(self.nor(self.conv1(image_vectorized)))
        image_vectorized = self.pool(image_vectorized)

        # image_vectorized = fc.relu(self.nor3(self.conv3(image_vectorized)))
        # image_vectorized = self.pool3(image_vectorized)

        # image_vectorized = fc.relu(self.nor4(self.conv4(image_vectorized)))
        # image_vectorized = self.pool4(image_vectorized)
        #
        # image_vectorized = fc.relu(self.nor4_5(self.conv4_5(image_vectorized)))

        # image_vectorized = fc.relu(self.nor5(self.conv5(image_vectorized)))
        # image_vectorized = self.pool5(image_vectorized)

        # image_vectorized = fc.relu(self.nor5_6(self.conv5_6(image_vectorized)))

        # image_vectorized = fc.relu(self.nor6(self.conv6(image_vectorized)))

        # image_vectorized = fc.relu(self.nor7(self.conv7(image_vectorized)))
        #
        # image_vectorized = fc.relu(self.nor8(self.conv8(image_vectorized)))
        # image_vectorized = self.pool8(image_vectorized)

        linear_output = image_vectorized.reshape(image_vectorized.shape[0], -1)
        # linear_output = self.dropout(linear_output)

        linear_output = fc.relu(self.linear(linear_output))

        # linear_output = fc.relu(self.linear3(linear_output))

        linear_output = self.linear4(linear_output)

        return linear_output
