
// lab4Dlg.h: 头文件
//

#pragma once


// Clab4Dlg 对话框
class Clab4Dlg : public CDialogEx
{
// 构造
public:
	Clab4Dlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_LAB4_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	CImage image, img_show;
	CRect rect;
	Mat img;
	Mat Element;
	Mat scaledImage, scaledImageOut;
	int cnt = 0, sum = 0;
	int op;
	//全局变量声明
//Mat SrcImg;
//颜色值变量
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	CEdit m_edit;
};
