with Ada.Text_IO;
procedure Ex1 is
   type My_Range is range 0 .. (2 ** 65 - 36893488147419103200);
   X : My_Range := My_Range'Last;
begin
   null;
end Ex1;
