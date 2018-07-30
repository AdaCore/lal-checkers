procedure Ex1 is
   A : Integer := 2;
   B : Integer := 3;
   C : Integer := 4;
   D : Integer := 5;
   S : Integer := 0;
begin
   for I in A + B .. C + D loop
      S := S + I;
   end loop;
end Ex1;
