
// lab4Dlg.cpp: 实现文件
//

#include "stdafx.h"
#include "lab4.h"
#include "lab4Dlg.h"
#include "afxdialogex.h"
#include "MatCImage.h"
#include "Translation.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// Clab4Dlg 对话框



Clab4Dlg::Clab4Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_LAB4_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Clab4Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT1, m_edit);
}

BEGIN_MESSAGE_MAP(Clab4Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &Clab4Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &Clab4Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &Clab4Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &Clab4Dlg::OnBnClickedButton4)
END_MESSAGE_MAP()


// Clab4Dlg 消息处理程序

BOOL Clab4Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标
	static CFont font;
	font.DeleteObject();
	font.CreatePointFont(220, _T("新宋体"));
	m_edit.SetFont(&font);//设置字体
	m_edit.SetWindowText(_T(""));
	// TODO: 在此添加额外的初始化代码
	CString title("车牌识别系统"); //在动态显示标题栏的标题内容
	this->SetWindowText(title);
	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void Clab4Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}



// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void Clab4Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		//CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();///!!!!!!!!如果注释掉就无法相应
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR Clab4Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void Clab4Dlg::OnBnClickedButton1()
{
		//清屏

		CEdit* pBoxOne;
		pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT1);
		pBoxOne->SetWindowText(_T(""));
		CRect rect1;
		GetDlgItem(IDC_ShowImg)->GetClientRect(&rect1);

		CWnd* pWnd = GetDlgItem(IDC_ShowImg);
		pWnd->GetClientRect(&rect1);//得到控件客户端区域坐标
		pWnd->ClientToScreen(rect1);//将区域坐标由 控件客户区转成对话框区
		//pWnd->GetWindowRect(&rect); //直接得到控件的对话框区坐标

		this->ScreenToClient(rect1); //将区域坐标由 对话框区转成对话框客户区坐标
		InvalidateRect(rect1);
		UpdateWindow();

		GetDlgItem(IDC_ShowOut)->GetClientRect(&rect1);
		pWnd = GetDlgItem(IDC_ShowOut);
		pWnd->GetClientRect(&rect1);//得到控件客户端区域坐标
		pWnd->ClientToScreen(rect1);//将区域坐标由 控件客户区转成对话框区
		//pWnd->GetWindowRect(&rect); //直接得到控件的对话框区坐标

		this->ScreenToClient(rect1); //将区域坐标由 对话框区转成对话框客户区坐标
		InvalidateRect(rect1);
		UpdateWindow();

		GetDlgItem(IDC_ShowImgfin)->GetClientRect(&rect1);
		pWnd = GetDlgItem(IDC_ShowImgfin);
		pWnd->GetClientRect(&rect1);//得到控件客户端区域坐标
		pWnd->ClientToScreen(rect1);//将区域坐标由 控件客户区转成对话框区
		//pWnd->GetWindowRect(&rect); //直接得到控件的对话框区坐标

		this->ScreenToClient(rect1); //将区域坐标由 对话框区转成对话框客户区坐标
		InvalidateRect(rect1);
		UpdateWindow();
		for (int i = 0; i < 7; i++) {
			CImage image1;
			if (i == 0) {
				GetDlgItem(IDC_Show1)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show1);
			}
			if (i == 1) {
				GetDlgItem(IDC_Show2)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show2);
			}
			if (i == 2) {
				GetDlgItem(IDC_Show3)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show3);
			}
			if (i == 3) {
				GetDlgItem(IDC_Show4)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show4);
			}
			if (i == 4) {
				GetDlgItem(IDC_Show5)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show5);
			}
			if (i == 5) {
				GetDlgItem(IDC_Show6)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show6);
			}
			if (i == 6) {
				GetDlgItem(IDC_Show7)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show7);
			}
			pWnd->GetClientRect(&rect1);//得到控件客户端区域坐标
			pWnd->ClientToScreen(rect1);//将区域坐标由 控件客户区转成对话框区
			//pWnd->GetWindowRect(&rect); //直接得到控件的对话框区坐标

			this->ScreenToClient(rect1); //将区域坐标由 对话框区转成对话框客户区坐标
			InvalidateRect(rect1);
			UpdateWindow();

			//ReleaseDC(pDC);//释放picture控件的DC 
		}
		// TODO: 在此添加控件通知处理程序代码

		// TODO: 在此添加控件通知处理程序代码
		//CString strFilter = _T("所有文件(*.*)|*.*|");
		//CFileDialog dlg(TRUE, NULL, NULL, NULL, strFilter, this);
		//SetDlgItemText(IDC_ImagePath, _T(" "));
		CFileDialog dlg(true, _T("*.bmp"), NULL, OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY,
			_T("image files (*.jpg ;*.bmp)|*.bmp;*.png;*.jpg |ALL Files (*.*) |*.*||"), NULL);

		//CFileDialog dlg(TRUE, _T("*.bmp;*.jpg;*.JPEG;*.JPG;*.tif;*.png"), NULL, OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY,
			//_T("image files All Files (*.*) |*.*||"), NULL);

		if (!dlg.DoModal() == IDOK)
			return;
		CString strFileName = dlg.GetPathName();
		CFileStatus status;
		if (!CFile::GetStatus(strFileName, status))
		{
			MessageBox(strFileName + "不存在", L"信息提示", MB_OK);
			return;
		}
		//HRESULT ret = image.Load(strFileName); // filename 是要加载的文件名（包含路径）

		//HBITMAP bitmap = image.Detach();
		CImage image1;//!!!!!!!!!!!!!!!不能用全局变量啊啊啊啊啊啊啊啊啊啊啊啊
		image1.Load(strFileName);

		//把CImage转化为Mat
		MatCImage matimage;
		matimage.CImageToMat(image1, img);
		///namedWindow("输入", WINDOW_NORMAL);
		//imshow("输入", img);
		//获取图片控件的大小
		//CRect rect1;
		GetDlgItem(IDC_ShowImg)->GetClientRect(&rect1);
		//width = pic_rect.right;
		//height = pic_rect.bottom;

		//改变图片大小适应picture控件
		resize(img, scaledImage, Size(rect1.Width(), rect1.Height()));

		//Mat转换为CImage
		if (!scaledImage.empty())
		{
			matimage.MatToCImage(scaledImage, image1);
		}
		op = 1;

		pWnd = GetDlgItem(IDC_ShowImg);
		CDC *pDC = pWnd->GetDC();
		if (!image1.IsNull())
		{
			image1.Draw(pDC->m_hDC, rect1); //将图片画到Picture控件表示的矩形区域    
		}

		ReleaseDC(pDC);//释放picture控件的DC  

}


void Clab4Dlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	//pBoxOne = (CEdit*)GetDlgItem(IDC_ResizeEDIT);
	sum++;
	Mat gray, result;

	//获取图片控件的大小
	CImage image1, imageOut;
	Mat lic, NumImg[7];
	char sss[7];
	string pro;
	string ssss;
	bool flag1 = 0, flag2 = 0;
	Recognition(img, lic,result, pro, sss, NumImg, flag1, flag2);
	//tiqu(result, result);
	if (flag1) ssss = "1";
	//imshow(ssss, img);
	if (flag1) {
		AfxMessageBox(TEXT("颜色提取车牌失败,重新选择图片"), MB_OK);
		flag2 = 1;
	}
	if (flag2) {
		if(flag1 == 0) AfxMessageBox(TEXT("字符提取失败,重新选择图片"), MB_OK);
		CString RESULT;
		CEdit* pBoxOne;
		RESULT.Format(_T("%d"), sum);
		pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT2);
		pBoxOne->SetWindowText(RESULT);

		RESULT.Format(_T("%d"), cnt);
		pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT3);
		pBoxOne->SetWindowText(RESULT);

		double per = cnt * 1.0 * 100 / sum;
		RESULT.Format(_T("%lf"), per);
		pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT4);
		pBoxOne->SetWindowText(RESULT);
	}
	else {
		CRect rect1;
		//GetDlgItem(IDC_ShowOut)->GetClientRect(&rect1);

		CWnd* pWnd = GetDlgItem(IDC_ShowOut);
		GetDlgItem(IDC_ShowOut)->GetClientRect(&rect1);
		resize(lic, scaledImageOut, Size(rect1.Width(), rect1.Height()));
		MatCImage matimage;
		//Mat转换为CImage
		if (!scaledImage.empty())
		{
			matimage.MatToCImage(scaledImageOut, imageOut);
		}
		//str.ReleaseBuffer();
		op = 2;
		pWnd = GetDlgItem(IDC_ShowOut);
		CDC *pDC = pWnd->GetDC();
		if (!imageOut.IsNull())
		{
			imageOut.Draw(pDC->m_hDC, rect1); //将图片画到Picture控件表示的矩形区域    
		}

		ReleaseDC(pDC);//释放picture控件的DC 

		//原图上显示
		GetDlgItem(IDC_ShowImgfin)->GetClientRect(&rect1);
		resize(result, scaledImageOut, Size(rect1.Width(), rect1.Height()));
		//MatCImage matimage;
		//Mat转换为CImage
		if (!scaledImage.empty())
		{
			matimage.MatToCImage(scaledImageOut, imageOut);
		}
		//str.ReleaseBuffer();
		pWnd = GetDlgItem(IDC_ShowImgfin);
		pDC = pWnd->GetDC();
		if (!imageOut.IsNull())
		{
			imageOut.Draw(pDC->m_hDC, rect1); //将图片画到Picture控件表示的矩形区域    
		}

		ReleaseDC(pDC);//释放picture控件的DC 
		for (int i = 0; i < 7; i++) {
			CImage image1;
			if (i == 0) {
				GetDlgItem(IDC_Show1)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show1);
			}
			if (i == 1) {
				GetDlgItem(IDC_Show2)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show2);
			}
			if (i == 2) {
				GetDlgItem(IDC_Show3)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show3);
			}
			if (i == 3) {
				GetDlgItem(IDC_Show4)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show4);
			}
			if (i == 4) {
				GetDlgItem(IDC_Show5)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show5);
			}
			if (i == 5) {
				GetDlgItem(IDC_Show6)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show6);
			}
			if (i == 6) {
				GetDlgItem(IDC_Show7)->GetClientRect(&rect1);
				pWnd = GetDlgItem(IDC_Show7);
			}
			//GetDlgItem(IDC_Show1)->GetClientRect(&rect1);
			resize(NumImg[i], scaledImageOut, Size(rect1.Width(), rect1.Height()));
			MatCImage matimage;
			//Mat转换为CImage
			if (!scaledImage.empty())
			{
				matimage.MatToCImage(scaledImageOut, image1);
			}
			//str.ReleaseBuffer();
			//pWnd = GetDlgItem(IDC_ShowOut);
			CDC *pDC = pWnd->GetDC();
			if (!image1.IsNull())
			{
				image1.Draw(pDC->m_hDC, rect1); //将图片画到Picture控件表示的矩形区域    
			}

			ReleaseDC(pDC);//释放picture控件的DC 
		}
		string str = "111111";
		for (int i = 0; i < 6; i++) str[i] = sss[i];
		string r = pro + str;
		CString RESULT;
		RESULT = r.c_str();
		//UpdateData(FALSE);
		CEdit* pBoxOne;
		pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT1);
		//pBoxOne->SetWindowText(RESULT);
		m_edit.SetWindowText(RESULT);
		MessageBox(TEXT("车牌字符识别成功"), MB_OK);
	}
}



void Clab4Dlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	cnt++;
	CString RESULT;
	CEdit* pBoxOne;
	RESULT.Format(_T("%d"), sum);
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT2);
	pBoxOne->SetWindowText(RESULT);

	RESULT.Format(_T("%d"), cnt);
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT3);
	pBoxOne->SetWindowText(RESULT);

	double per = cnt * 1.0 * 100 / sum;
	RESULT.Format(_T("%lf"), per);
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT4);
	pBoxOne->SetWindowText(RESULT);
}


void Clab4Dlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	CString RESULT;
	CEdit* pBoxOne;
	RESULT.Format(_T("%d"), sum);
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT2);
	pBoxOne->SetWindowText(RESULT);

	RESULT.Format(_T("%d"), cnt);
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT3);
	pBoxOne->SetWindowText(RESULT);

	double per = cnt * 1.0 * 100 / sum;
	RESULT.Format(_T("%lf"), per);
	pBoxOne = (CEdit*)GetDlgItem(IDC_EDIT4);
	pBoxOne->SetWindowText(RESULT);
}
