import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, infilter, outfilter): # 어차피 stride는 무조건 1이다. 업샘플링 뒤에서 따로 할거라서
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(infilter, outfilter, 3, 1, padding = 1),
            nn.BatchNorm2d(outfilter),
            nn.PReLU(),
            nn.Conv2d(outfilter, outfilter, 3, 1, padding =1),
            nn.BatchNorm2d(outfilter)
        )
    def forward(self ,x):
        return self.conv(x) + x


# residual 뒤에 액티베이션 적용 안한다.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 ,64 ,9 ,1 ,padding = 4),
            nn.PReLU()
        )
        # Residual 짜기 너무 힘들다..
        # Block class 지정하고 이 클래스를 이용해서 make layer 함수로 만든다.
        self.basic_block = self._make_layer(ResidualBlock ,5 ,64 ,64)

        self.basic_block2 = nn.Sequential(
            nn.Conv2d(64 ,64 ,3 ,1 ,padding=1),
            nn.BatchNorm2d(64)
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(64 ,256 ,3 ,1 ,padding=1), # pixelshuffler로 가로 세로 2배씩 늘릴거야
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64 ,256 ,3 ,1 ,padding=1), # pixelshuffler로 가로 세로 2배씩 늘릴거야
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64 ,3 ,9 ,1 ,padding=4),
            nn.Tanh()
        )

    # 아마 여기에서 구조적 오류가 있는듯 싶다. 동일한 레이어를 계속 써댔으니..
    def forward(self ,x):
        x = self.conv1(x)

        all_skip = x

        x = self.basic_block(x)
        x = self.basic_block2(x)
        x = torch.add(x, all_skip)

        x = self.last_block(x)

        return x

    def _make_layer(self, resblock, num_block ,infilter ,outfilter):
        layers = []
        for _ in range(num_block):
            layers.append(resblock(infilter, outfilter))
            infilter = outfilter

        return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 ,64 ,3 ,1 ,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.basic_blocks_1 = nn.Sequential(
            nn.Conv2d(64 ,64 ,3 ,2 ,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.basic_blocks_2 = nn.Sequential(
            nn.Conv2d(64 ,128 ,3 ,1 ,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 ,128 ,3 ,2 ,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 ,256 ,3 ,1 ,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 ,256 ,3 ,2 ,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 ,512 ,3 ,1 ,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512 ,512 ,3 ,2 ,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2) # 24*24*512
        )
        self.fcs = nn.Sequential(
            nn.Linear(24 *24 *512 ,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024 ,1),
            nn.Sigmoid()
        )
    def forward(self ,x):
        x = self.conv1(x)
        content = self.basic_blocks_1(x)
        x = self.basic_blocks_2(content)
        x = x.view(-1 ,24 *24 *512)
        x = self.fcs(x)
        return x ,content